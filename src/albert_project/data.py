from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class TaskSpec:
    name: str
    hf_name: str
    text_columns: tuple[str, ...]
    num_labels: int
    metric_names: tuple[str, ...]
    train_split: str = "train"
    validation_split: str = "validation"


GLUE_DATASET_ID = "nyu-mll/glue"


TASKS: dict[str, TaskSpec] = {
    "sst2": TaskSpec(
        name="sst2",
        hf_name="sst2",
        text_columns=("sentence",),
        num_labels=2,
        metric_names=("accuracy",),
    ),
    "mrpc": TaskSpec(
        name="mrpc",
        hf_name="mrpc",
        text_columns=("sentence1", "sentence2"),
        num_labels=2,
        metric_names=("accuracy", "f1"),
    ),
    "rte": TaskSpec(
        name="rte",
        hf_name="rte",
        text_columns=("sentence1", "sentence2"),
        num_labels=2,
        metric_names=("accuracy",),
    ),
    "mnli": TaskSpec(
        name="mnli",
        hf_name="mnli",
        text_columns=("premise", "hypothesis"),
        num_labels=3,
        metric_names=("accuracy",),
        validation_split="validation_matched",
    ),
    "mnli-mm": TaskSpec(
        name="mnli-mm",
        hf_name="mnli",
        text_columns=("premise", "hypothesis"),
        num_labels=3,
        metric_names=("accuracy",),
        validation_split="validation_mismatched",
    ),
}


TASK_ALIASES = {
    "sst2": "sst2",
    "sst-2": "sst2",
    "mrpc": "mrpc",
    "rte": "rte",
    "mnli": "mnli",
    "mnli-m": "mnli",
    "mnli-matched": "mnli",
    "mnli-mm": "mnli-mm",
    "mnli-mismatched": "mnli-mm",
}


def normalize_task_name(task_name: str) -> str:
    normalized = task_name.strip().lower().replace("_", "-")
    return TASK_ALIASES.get(normalized, normalized)


def get_task_spec(task_name: str) -> TaskSpec:
    normalized = normalize_task_name(task_name)
    if normalized not in TASKS:
        raise ValueError(
            f"Unsupported task '{task_name}'. Supported tasks: {', '.join(sorted(TASKS))}."
        )
    return TASKS[normalized]


def load_glue_task(task_name: str) -> DatasetDict:
    """Load a supported GLUE task from Hugging Face datasets."""
    spec = get_task_spec(task_name)
    return load_dataset(GLUE_DATASET_ID, spec.hf_name)


def select_train_fraction(dataset_dict: DatasetDict, fraction: float, seed: int) -> DatasetDict:
    """Return a copy with only a fraction of the train split.

    The validation split is intentionally kept unchanged so low-data extension
    results remain comparable across fractions.
    """
    if not (0 < fraction <= 1):
        raise ValueError("train_fraction must be in the interval (0, 1].")

    if fraction == 1:
        return dataset_dict

    train = dataset_dict["train"]
    subset_size = max(1, int(len(train) * fraction))
    shuffled = train.shuffle(seed=seed)

    dataset_dict = DatasetDict(dataset_dict)
    dataset_dict["train"] = shuffled.select(range(subset_size))
    return dataset_dict


def limit_dataset_splits(
    dataset_dict: DatasetDict,
    *,
    seed: int,
    train_split: str = "train",
    validation_split: str = "validation",
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
) -> DatasetDict:
    """Optionally limit train/eval splits for quick smoke tests.

    Full experiments should leave these values unset. They are useful for
    verifying the pipeline before launching long fine-tuning runs.
    """
    dataset_dict = DatasetDict(dataset_dict)

    if max_train_samples is not None:
        if max_train_samples <= 0:
            raise ValueError("max_train_samples must be positive when provided.")
        train = dataset_dict[train_split].shuffle(seed=seed)
        dataset_dict[train_split] = train.select(range(min(max_train_samples, len(train))))

    if max_eval_samples is not None:
        if max_eval_samples <= 0:
            raise ValueError("max_eval_samples must be positive when provided.")
        validation = dataset_dict[validation_split].shuffle(seed=seed)
        dataset_dict[validation_split] = validation.select(
            range(min(max_eval_samples, len(validation)))
        )

    return dataset_dict


def tokenize_dataset(
    dataset_dict: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    task_name: str,
    max_length: int,
) -> DatasetDict:
    """Tokenize splits for supported GLUE tasks."""
    spec = get_task_spec(task_name)

    def preprocess(batch: dict[str, Any]) -> dict[str, Any]:
        if len(spec.text_columns) == 1:
            return tokenizer(
                batch[spec.text_columns[0]],
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )

        return tokenizer(
            batch[spec.text_columns[0]],
            batch[spec.text_columns[1]],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized = dataset_dict.map(preprocess, batched=True)

    columns_to_keep = {"input_ids", "attention_mask", "token_type_ids", "label"}
    for split in tokenized.keys():
        remove_columns = [
            column
            for column in tokenized[split].column_names
            if column not in columns_to_keep
        ]
        if remove_columns:
            tokenized[split] = tokenized[split].remove_columns(remove_columns)

    return tokenized