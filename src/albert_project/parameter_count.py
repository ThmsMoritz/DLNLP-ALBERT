from __future__ import annotations

from typing import Any

from .data import get_task_spec
from .modeling import count_embedding_parameters, count_parameters, load_model_and_tokenizer


def parameter_summary(config: dict[str, Any]) -> dict[str, Any]:
    task_name = config.get("task_name", "sst2")
    spec = get_task_spec(task_name)
    loaded = load_model_and_tokenizer(config, num_labels=spec.num_labels)
    return {
        "experiment_name": config.get("experiment_name", "parameter_count"),
        "configuration": config.get("configuration", loaded.model_label),
        "model_label": loaded.model_label,
        "model_name": config.get("model_name", "from_config"),
        "model_source": config.get("model_source", "pretrained"),
        "task_name": task_name,
        "parameters": count_parameters(loaded.model),
        "embedding_parameters": count_embedding_parameters(loaded.model),
        "notes": loaded.notes or config.get("notes", ""),
    }
