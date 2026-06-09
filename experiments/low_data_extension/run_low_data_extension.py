from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from albert_project.config import load_yaml
from albert_project.results import append_result_csv
from albert_project.training import run_finetuning_experiment


DEFAULT_CONFIG = "configs/low_data_extension/low_data_sst2.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run low-data extension experiment.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--results", default="results/low_data_results.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = load_yaml(PROJECT_ROOT / args.config)
    models = base_config.get("models", [])
    fractions = base_config.get("train_fractions", [0.1, 0.25, 0.5, 1.0])

    if not models:
        raise ValueError("Low-data config must define a non-empty 'models' list.")

    for model_config in models:
        for fraction in fractions:
            config = deepcopy(base_config)
            config.pop("models", None)
            config.pop("train_fractions", None)
            config.update(model_config)
            config["train_fraction"] = float(fraction)
            config["experiment_name"] = "low_data_extension"
            config["configuration"] = f"{model_config.get('model_label', model_config.get('model_name'))}_{fraction}"
            print(f"\n=== Low-data run: {config['configuration']} ===")
            result = run_finetuning_experiment(config)
            append_result_csv(PROJECT_ROOT / args.results, result)
            print(result)


if __name__ == "__main__":
    main()
