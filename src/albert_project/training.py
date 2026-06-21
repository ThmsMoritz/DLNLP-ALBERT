from __future__ import annotations

import inspect
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

from .data import (
    get_task_spec,
    limit_dataset_splits,
    load_glue_task,
    select_train_fraction,
    tokenize_dataset,
)
from .modeling import count_embedding_parameters, count_parameters, load_model_and_tokenizer


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    eval_batch_size: int,
    learning_rate: float,
    seed: int,
    weight_decay: float,
    warmup_ratio: float,
    gradient_accumulation_steps: int,
    logging_steps: int,
    save_model: bool,
    fp16: bool,
) -> TrainingArguments:
    """Create TrainingArguments with compatibility across Transformers versions."""
    common = dict(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=logging_steps,
        save_strategy="epoch" if save_model else "no",
        report_to="none",
        seed=seed,
        data_seed=seed,
        fp16=fp16,
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
    set_global_seed(seed)

    train_fraction = float(config.get("train_fraction", 1.0))
    max_train_samples = config.get("max_train_samples")
    max_eval_samples = config.get("max_eval_samples")
    max_train_samples = int(max_train_samples) if max_train_samples is not None else None
    max_eval_samples = int(max_eval_samples) if max_eval_samples is not None else None

    max_length = int(config.get("max_length", 128))
    epochs = float(config.get("epochs", 3))
    batch_size = int(config.get("batch_size", 16))
    eval_batch_size = int(config.get("eval_batch_size", batch_size))
    learning_rate = float(config.get("learning_rate", 2e-5))
    weight_decay = float(config.get("weight_decay", 0.0))
    warmup_ratio = float(config.get("warmup_ratio", 0.0))
    gradient_accumulation_steps = int(config.get("gradient_accumulation_steps", 1))
    logging_steps = int(config.get("logging_steps", 50))
    save_model = bool(config.get("save_model", False))
    fp16 = bool(config.get("fp16", False))

    loaded = load_model_and_tokenizer(config, num_labels=spec.num_labels)
    model = loaded.model
    tokenizer = loaded.tokenizer

    raw_dataset = load_glue_task(task_name)
    raw_dataset = select_train_fraction(raw_dataset, fraction=train_fraction, seed=seed)
    raw_dataset = limit_dataset_splits(
        raw_dataset,
        seed=seed,
        train_split=spec.train_split,
        validation_split=spec.validation_split,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
    )
    tokenized = tokenize_dataset(raw_dataset, tokenizer, task_name=task_name, max_length=max_length)

    output_dir = config.get(
        "output_dir",
        f"outputs/{config.get('experiment_name', 'experiment')}/{loaded.model_label}/{spec.name}/{train_fraction}",
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    args = _build_training_args(
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        seed=seed,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=logging_steps,
        save_model=save_model,
        fp16=fp16,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=tokenized[spec.train_split],
        eval_dataset=tokenized[spec.validation_split],
        data_collator=data_collator,
        compute_metrics=compute_metrics_for_task(task_name),
    )

    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

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
        "task_name": spec.name,
        "hf_task_name": spec.hf_name,
        "train_split": spec.train_split,
        "validation_split": spec.validation_split,
        "train_fraction": train_fraction,
        "max_train_samples": max_train_samples or "",
        "max_eval_samples": max_eval_samples or "",
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "max_length": max_length,
        "train_examples": len(tokenized[spec.train_split]),
        "validation_examples": len(tokenized[spec.validation_split]),
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
