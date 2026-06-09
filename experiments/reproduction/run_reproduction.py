from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from albert_project.config import load_yaml
from albert_project.results import append_result_csv
from albert_project.training import run_finetuning_experiment


DEFAULT_CONFIGS = [
    "configs/reproduction/bert_base_sst2.yaml",
    "configs/reproduction/bert_base_mrpc.yaml",
    "configs/reproduction/albert_base_sst2.yaml",
    "configs/reproduction/albert_base_mrpc.yaml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BERT-base vs ALBERT-base reproduction experiments.")
    parser.add_argument("--config", action="append", help="Path to one config. Can be repeated.")
    parser.add_argument("--results", default="results/reproduction_results.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_paths = args.config or DEFAULT_CONFIGS
    for config_path in config_paths:
        print(f"\n=== Running reproduction config: {config_path} ===")
        config = load_yaml(PROJECT_ROOT / config_path)
        result = run_finetuning_experiment(config)
        append_result_csv(PROJECT_ROOT / args.results, result)
        print(result)


if __name__ == "__main__":
    main()
