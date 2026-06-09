from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from datasets import DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class TaskSpec:
    name: str
    hf_name: str
    text_columns: tuple[str, ...]
    num_labels: int
    metric_names: tuple[str, ...]


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
}


def get_task_spec(task_name: str) -> TaskSpec:
    normalized = task_name.lower().replace("-", "")
    if normalized == "sst2":
        normalized = "sst2"
    if normalized not in TASKS:
        raise ValueError(
            f"Unsupported task '{task_name}'. Supported tasks: {', '.join(sorted(TASKS))}."
        )
    return TASKS[normalized]


def load_glue_task(task_name: str) -> DatasetDict:
    """Load a GLUE task from Hugging Face datasets."""
    spec = get_task_spec(task_name)
    return load_dataset("glue", spec.hf_name)


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


def tokenize_dataset(
    dataset_dict: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    task_name: str,
    max_length: int,
) -> DatasetDict:
    """Tokenize train and validation splits for SST-2 or MRPC."""
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
        remove_columns = [c for c in tokenized[split].column_names if c not in columns_to_keep]
        if remove_columns:
            tokenized[split] = tokenized[split].remove_columns(remove_columns)
    return tokenized
