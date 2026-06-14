from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import DatasetDict
from sklearn.metrics import accuracy_score
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from albert_project.config import load_yaml
from albert_project.data import get_task_spec, load_glue_task, tokenize_dataset
from albert_project.modeling import (
    count_embedding_parameters,
    count_parameters,
    load_model_and_tokenizer,
)
from albert_project.results import append_result_csv


CONFIG_PATH = PROJECT_ROOT / "preliminary_tests" / "parameter_sharing_smoke.yaml"
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "results" / "preliminary_tests" / "parameter_sharing_smoke_results.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny parameter-sharing smoke test.")
    parser.add_argument("--config", default=str(CONFIG_PATH), help="Path to a smoke-test YAML config.")
    parser.add_argument(
        "--results",
        default=str(DEFAULT_RESULTS_PATH),
        help="Path to the preliminary smoke-test result CSV.",
    )
    return parser.parse_args()


def limit_split(dataset_dict: DatasetDict, split: str, max_samples: int) -> None:
    sample_count = min(max_samples, len(dataset_dict[split]))
    dataset_dict[split] = dataset_dict[split].select(range(sample_count))


def compute_accuracy(eval_pred: Any) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": float(accuracy_score(labels, predictions))}


def build_training_args(output_dir: Path, config: dict[str, Any]) -> TrainingArguments:
    common = dict(
        output_dir=str(output_dir),
        num_train_epochs=float(config.get("epochs", 1)),
        per_device_train_batch_size=int(config.get("batch_size", 4)),
        per_device_eval_batch_size=int(config.get("batch_size", 4)),
        learning_rate=float(config.get("learning_rate", 2e-5)),
        save_strategy="no",
        report_to="none",
        seed=int(config.get("seed", 42)),
    )
    try:
        return TrainingArguments(eval_strategy="epoch", **common)
    except TypeError:
        return TrainingArguments(evaluation_strategy="epoch", **common)


def main() -> None:
    cli_args = parse_args()
    config_path = Path(cli_args.config)
    config = load_yaml(PROJECT_ROOT / config_path if not config_path.is_absolute() else config_path)
    task_name = config.get("task_name", "sst2")
    spec = get_task_spec(task_name)
    loaded = load_model_and_tokenizer(config, num_labels=spec.num_labels)

    raw_dataset = DatasetDict(load_glue_task(task_name))
    limit_split(raw_dataset, "train", int(config.get("max_train_samples", 32)))
    limit_split(raw_dataset, "validation", int(config.get("max_eval_samples", 32)))
    tokenized = tokenize_dataset(
        raw_dataset,
        loaded.tokenizer,
        task_name=task_name,
        max_length=int(config.get("max_length", 64)),
    )

    output_dir = PROJECT_ROOT / "outputs" / "preliminary_tests" / config["configuration"]
    training_args = build_training_args(output_dir, config)
    trainer_kwargs = dict(
        model=loaded.model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorWithPadding(tokenizer=loaded.tokenizer),
        compute_metrics=compute_accuracy,
    )
    try:
        trainer = Trainer(processing_class=loaded.tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = Trainer(tokenizer=loaded.tokenizer, **trainer_kwargs)

    start = time.perf_counter()
    trainer.train()
    training_time_seconds = time.perf_counter() - start
    metrics = trainer.evaluate()

    result = {
        "experiment_name": config["experiment_name"],
        "configuration": config["configuration"],
        "model_label": loaded.model_label,
        "task_name": task_name,
        "sharing_strategy": config["sharing_strategy"],
        "train_examples": len(tokenized["train"]),
        "validation_examples": len(tokenized["validation"]),
        "parameters": count_parameters(loaded.model),
        "embedding_parameters": count_embedding_parameters(loaded.model),
        "training_time_seconds": round(training_time_seconds, 3),
        "gpu_available": torch.cuda.is_available(),
        "notes": "Preliminary smoke test only; not a final experiment result.",
    }
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            result[key.replace("eval_", "")] = float(value)

    results_path = Path(cli_args.results)
    append_result_csv(PROJECT_ROOT / results_path if not results_path.is_absolute() else results_path, result)
    print(result)


if __name__ == "__main__":
    main()
