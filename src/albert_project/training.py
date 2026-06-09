from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

from .data import get_task_spec, load_glue_task, select_train_fraction, tokenize_dataset
from .modeling import count_embedding_parameters, count_parameters, load_model_and_tokenizer


def compute_metrics_for_task(task_name: str):
    spec = get_task_spec(task_name)

    def compute(eval_pred: Any) -> dict[str, float]:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        result = {"accuracy": float(accuracy_score(labels, predictions))}
        if "f1" in spec.metric_names:
            result["f1"] = float(f1_score(labels, predictions))
        return result

    return compute


def _build_training_args(
    output_dir: str,
    epochs: float,
    batch_size: int,
    learning_rate: float,
    seed: int,
    weight_decay: float,
    logging_steps: int,
    save_model: bool,
) -> TrainingArguments:
    """Create TrainingArguments with compatibility across Transformers versions."""
    common = dict(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        save_strategy="epoch" if save_model else "no",
        report_to="none",
        seed=seed,
        load_best_model_at_end=False,
    )
    try:
        return TrainingArguments(eval_strategy="epoch", **common)
    except TypeError:
        return TrainingArguments(evaluation_strategy="epoch", **common)


def run_finetuning_experiment(config: dict[str, Any]) -> dict[str, Any]:
    """Run one fine-tuning experiment and return a flat result dictionary."""
    task_name = config.get("task_name", "sst2")
    spec = get_task_spec(task_name)
    seed = int(config.get("seed", 42))
    train_fraction = float(config.get("train_fraction", 1.0))
    max_length = int(config.get("max_length", 128))
    epochs = float(config.get("epochs", 3))
    batch_size = int(config.get("batch_size", 16))
    learning_rate = float(config.get("learning_rate", 2e-5))
    weight_decay = float(config.get("weight_decay", 0.0))
    logging_steps = int(config.get("logging_steps", 50))
    save_model = bool(config.get("save_model", False))

    loaded = load_model_and_tokenizer(config, num_labels=spec.num_labels)
    model = loaded.model
    tokenizer = loaded.tokenizer

    raw_dataset = load_glue_task(task_name)
    raw_dataset = select_train_fraction(raw_dataset, fraction=train_fraction, seed=seed)
    tokenized = tokenize_dataset(raw_dataset, tokenizer, task_name=task_name, max_length=max_length)

    output_dir = config.get(
        "output_dir",
        f"outputs/{config.get('experiment_name', 'experiment')}/{loaded.model_label}/{task_name}/{train_fraction}",
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    args = _build_training_args(
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        save_model=save_model,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_for_task(task_name),
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    trainer.train()
    training_time_seconds = time.perf_counter() - start
    metrics = trainer.evaluate()

    gpu_memory_mb = None
    if torch.cuda.is_available():
        gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)

    result = {
        "experiment_name": config.get("experiment_name", "experiment"),
        "configuration": config.get("configuration", loaded.model_label),
        "model_label": loaded.model_label,
        "model_name": config.get("model_name", "from_config"),
        "model_source": config.get("model_source", "pretrained"),
        "task_name": task_name,
        "train_fraction": train_fraction,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "train_examples": len(tokenized["train"]),
        "validation_examples": len(tokenized["validation"]),
        "parameters": count_parameters(model),
        "embedding_parameters": count_embedding_parameters(model),
        "training_time_seconds": round(training_time_seconds, 3),
        "gpu_memory_mb": round(gpu_memory_mb, 3) if gpu_memory_mb is not None else "NA",
        "notes": loaded.notes or config.get("notes", ""),
    }

    for key, value in metrics.items():
        cleaned_key = key.replace("eval_", "")
        if isinstance(value, (int, float)):
            result[cleaned_key] = float(value)

    return result
