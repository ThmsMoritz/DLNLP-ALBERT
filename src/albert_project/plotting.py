from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def plot_reproduction_results(csv_path: str, output_dir: str = "plots") -> None:
    df = pd.read_csv(csv_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if "parameters" in df.columns:
        df["parameters"] = _numeric_column(df, "parameters")
        param_df = df.dropna(subset=["parameters"]).drop_duplicates("model_label")[
            ["model_label", "parameters"]
        ]
        if not param_df.empty:
            plt.figure()
            plt.bar(param_df["model_label"], param_df["parameters"])
            plt.ylabel("Trainable parameters")
            plt.title("BERT vs ALBERT parameter count")
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            plt.savefig(Path(output_dir) / "bert_vs_albert_parameters.png")
            plt.close()

    if "accuracy" in df.columns:
        df["accuracy"] = _numeric_column(df, "accuracy")
        perf_df = df.dropna(subset=["accuracy"])
        if not perf_df.empty:
            perf = perf_df.pivot_table(index="task_name", columns="model_label", values="accuracy")
            perf.plot(kind="bar")
            plt.ylabel("Validation accuracy")
            plt.title("BERT vs ALBERT task performance")
            plt.xticks(rotation=0)
            plt.tight_layout()
            plt.savefig(Path(output_dir) / "bert_vs_albert_accuracy.png")
            plt.close()

    if "f1" in df.columns:
        df["f1"] = _numeric_column(df, "f1")
        f1_df = df.dropna(subset=["f1"])
        if not f1_df.empty:
            f1_perf = f1_df.pivot_table(index="task_name", columns="model_label", values="f1")
            f1_perf.plot(kind="bar")
            plt.ylabel("Validation F1")
            plt.title("BERT vs ALBERT F1 performance")
            plt.xticks(rotation=0)
            plt.tight_layout()
            plt.savefig(Path(output_dir) / "bert_vs_albert_f1.png")
            plt.close()


def plot_ablation_results(csv_path: str, output_prefix: str, output_dir: str = "plots") -> None:
    df = pd.read_csv(csv_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if "accuracy" in df.columns:
        df["accuracy"] = _numeric_column(df, "accuracy")
        perf_df = df.dropna(subset=["accuracy"])
        if not perf_df.empty:
            perf = perf_df.pivot_table(index="configuration", columns="task_name", values="accuracy")
            perf.plot(kind="bar")
            plt.ylabel("Validation accuracy")
            plt.title(f"{output_prefix.replace('_', ' ').title()} performance")
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            plt.savefig(Path(output_dir) / f"{output_prefix}_performance.png")
            plt.close()

    if "parameters" in df.columns:
        df["parameters"] = _numeric_column(df, "parameters")
        param_df = df.dropna(subset=["parameters"]).drop_duplicates("configuration")[
            ["configuration", "parameters"]
        ]
        if not param_df.empty:
            plt.figure()
            plt.bar(param_df["configuration"], param_df["parameters"])
            plt.ylabel("Trainable parameters")
            plt.title(f"{output_prefix.replace('_', ' ').title()} parameter count")
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            plt.savefig(Path(output_dir) / f"{output_prefix}_parameters.png")
            plt.close()
