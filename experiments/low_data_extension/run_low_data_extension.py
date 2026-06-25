from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from albert_project.config import load_yaml
from albert_project.results import append_result_csv


DEFAULT_CONFIG = "configs/low_data_extension/low_data_matrix.yaml"
MATRIX_KEYS = {"models", "tasks", "train_fractions", "common"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the low-data extension experiments.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--results", default="results/low_data_results.csv")
    parser.add_argument(
        "--only-task",
        action="append",
        help="Run only this task name. Can be repeated, e.g. --only-task sst2 --only-task rte.",
    )
    parser.add_argument(
        "--only-model",
        action="append",
        help="Run only models whose normalized model_label or model_name exactly matches or starts with this value.",
    )
    parser.add_argument(
        "--only-fraction",
        action="append",
        type=float,
        help="Run only this training fraction. Can be repeated, e.g. --only-fraction 0.1 --only-fraction 0.25.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Override configs with tiny train/eval subsets to verify that the pipeline works.",
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="Print the expanded run configurations without starting training.",
    )
    return parser.parse_args()


def _as_dict(item: str | dict[str, Any], key: str) -> dict[str, Any]:
    if isinstance(item, dict):
        return deepcopy(item)
    return {key: item}


def _fraction_label(fraction: float) -> str:
    return f"{int(round(fraction * 100))}pct"


def _safe_config_name(value: str) -> str:
    return value.replace("/", "-").replace(" ", "-")


def expand_config(config: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Expand a model x task x train-fraction matrix into single run configs."""
    if "models" not in config:
        single_config = deepcopy(config)
        single_config.setdefault("train_fraction", float(single_config.get("train_fraction", 1.0)))
        yield single_config
        return

    common = deepcopy(config.get("common", {}))
    top_level_defaults = {k: deepcopy(v) for k, v in config.items() if k not in MATRIX_KEYS}
    common = {**top_level_defaults, **common}
    fractions = config.get("train_fractions", [0.10, 0.25, 0.50])

    tasks = config.get("tasks")
    if tasks is None:
        if "task_name" not in common:
            raise ValueError("Low-data matrix config must define either 'tasks' or a top-level 'task_name'.")
        tasks = [{"task_name": common["task_name"]}]

    for model_config_raw in config["models"]:
        model_config = _as_dict(model_config_raw, "model_name")
        for task_config_raw in tasks:
            task_config = _as_dict(task_config_raw, "task_name")
            for fraction in fractions:
                train_fraction = float(fraction)
                run_config = deepcopy(common)
                run_config.update(task_config)
                run_config.update(model_config)
                run_config["train_fraction"] = train_fraction

                model_label = run_config.get("model_label", run_config.get("model_name", "model"))
                task_name = run_config.get("task_name", "task")
                run_config.setdefault("experiment_name", "low_data_extension")
                run_config.setdefault(
                    "configuration",
                    _safe_config_name(f"{model_label}_{task_name}_{_fraction_label(train_fraction)}"),
                )
                yield run_config


def _normalize_task_filter(task_name: str) -> str:
    normalized = task_name.strip().lower().replace("_", "-")
    aliases = {"sst-2": "sst2", "mnli-m": "mnli", "mnli-matched": "mnli"}
    return aliases.get(normalized, normalized)


def _normalize_model_filter(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def _model_matches(config: dict[str, Any], model_filter: str) -> bool:
    requested = _normalize_model_filter(model_filter)
    model_label = _normalize_model_filter(str(config.get("model_label", "")))
    model_name = _normalize_model_filter(str(config.get("model_name", "")))
    candidates = {model_label, model_name}
    return requested in candidates or any(candidate.startswith(f"{requested}-") for candidate in candidates)


def _fraction_matches(config: dict[str, Any], allowed_fractions: list[float]) -> bool:
    current = float(config.get("train_fraction", 1.0))
    return any(abs(current - float(fraction)) < 1e-9 for fraction in allowed_fractions)


def should_run(
    config: dict[str, Any],
    only_tasks: list[str] | None,
    only_models: list[str] | None,
    only_fractions: list[float] | None,
) -> bool:
    if only_tasks:
        allowed_tasks = {_normalize_task_filter(t) for t in only_tasks}
        task_name = _normalize_task_filter(str(config.get("task_name", "")))
        if task_name not in allowed_tasks:
            return False

    if only_models and not any(_model_matches(config, model_filter) for model_filter in only_models):
        return False

    if only_fractions and not _fraction_matches(config, only_fractions):
        return False

    return True


def apply_smoke_test_overrides(config: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(config)
    config["epochs"] = 1
    config["max_train_samples"] = 32
    config["max_eval_samples"] = 32
    config["logging_steps"] = 5
    config["configuration"] = f"smoke_{config.get('configuration', 'run')}"
    config["notes"] = "Smoke-test subset; do not use for final reported results."
    return config


def main() -> None:
    args = parse_args()
    loaded_config = load_yaml(PROJECT_ROOT / args.config)

    for config in expand_config(loaded_config):
        if args.smoke_test:
            config = apply_smoke_test_overrides(config)
        if not should_run(config, args.only_task, args.only_model, args.only_fraction):
            continue

        if args.list_configs:
            print(
                f"{config['configuration']}: "
                f"model={config.get('model_name')} "
                f"task={config.get('task_name')} "
                f"train_fraction={config.get('train_fraction')}"
            )
            continue

        from albert_project.training import run_finetuning_experiment

        print(f"\n=== Running low-data config: {config['configuration']} ===")
        result = run_finetuning_experiment(config)
        append_result_csv(PROJECT_ROOT / args.results, result)
        print(result)


if __name__ == "__main__":
    main()
