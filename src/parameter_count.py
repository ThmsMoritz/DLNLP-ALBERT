from __future__ import annotations

import argparse
from pathlib import Path

from albert_project.config import load_yaml
from albert_project.parameter_count import parameter_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Print parameter count for one config.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    summary = parameter_summary(load_yaml(Path(args.config)))
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
