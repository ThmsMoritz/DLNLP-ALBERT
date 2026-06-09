from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import pandas as pd


def append_result_csv(path: str | Path, result: dict) -> None:
    """Append one result row to a CSV, creating headers if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()

    if file_exists:
        existing = pd.read_csv(path)
        columns = list(existing.columns)
        for key in result.keys():
            if key not in columns:
                columns.append(key)
        row = {column: result.get(column, "") for column in columns}
        updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
        updated.to_csv(path, index=False)
        return

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(result.keys()))
        writer.writeheader()
        writer.writerow(result)


def write_results_csv(path: str | Path, results: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    results = list(results)
    if not results:
        return
    keys: list[str] = []
    for row in results:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key, "") for key in keys})
