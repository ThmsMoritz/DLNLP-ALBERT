from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from albert_project.config import load_yaml
from albert_project.parameter_count import parameter_summary
from albert_project.results import append_result_csv
from albert_project.training import run_finetuning_experiment


DEFAULT_CONFIGS = [
    "configs/factorized_embeddings/standard_embeddings.yaml",
    "configs/factorized_embeddings/factorized_embeddings.yaml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run factorized embedding ablation.")
    parser.add_argument("--config", action="append", help="Path to one config. Can be repeated.")
    parser.add_argument("--results", default="results/embedding_results.csv")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Actually fine-tune from config. Without this flag only parameter counts are produced.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_paths = args.config or DEFAULT_CONFIGS
    for config_path in config_paths:
        print(f"\n=== Embedding ablation config: {config_path} ===")
        config = load_yaml(PROJECT_ROOT / config_path)
        result = run_finetuning_experiment(config) if args.train else parameter_summary(config)
        append_result_csv(PROJECT_ROOT / args.results, result)
        print(result)


if __name__ == "__main__":
    main()
