from __future__ import annotations

from pathlib import Path

from albert_project.plotting import plot_ablation_results, plot_reproduction_results


if __name__ == "__main__":
    if Path("results/reproduction_results.csv").exists():
        plot_reproduction_results("results/reproduction_results.csv")
    if Path("results/parameter_sharing_results.csv").exists():
        plot_ablation_results("results/parameter_sharing_results.csv", "parameter_sharing")
    if Path("results/embedding_results.csv").exists():
        plot_ablation_results("results/embedding_results.csv", "embedding_ablation")
    if Path("results/low_data_results.csv").exists():
        plot_ablation_results("results/low_data_results.csv", "low_data")
