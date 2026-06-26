from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results" / "parameter_sharing_results.csv"
PLOTS = PROJECT_ROOT / "plots"

PLOTS.mkdir(exist_ok=True)

# ------------------------------------------------------------------
# Load results
# ------------------------------------------------------------------

df = pd.read_csv(RESULTS)

# Keep only SST-2 results (parameter counts are identical for MRPC)
df = df[df["task_name"] == "sst2"].copy()

# Desired plotting order
order = [
    "no_sharing",
    "shared_attention",
    "shared_ffn",
    "lower_half_sharing",
    "upper_half_sharing",
    "full_sharing",
]

labels = {
    "no_sharing": "No Sharing",
    "shared_attention": "Shared\nAttention",
    "shared_ffn": "Shared\nFFN",
    "lower_half_sharing": "Lower\nHalf",
    "upper_half_sharing": "Upper\nHalf",
    "full_sharing": "Full\nSharing",
}

df["configuration"] = pd.Categorical(
    df["configuration"],
    categories=order,
    ordered=True,
)

df = df.sort_values("configuration")

# Convert to millions
df["parameters_m"] = df["parameters"] / 1_000_000

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------

plt.style.use("ggplot")

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    [labels[c] for c in df["configuration"]],
    df["parameters_m"],
)

ax.set_title("Parameter Count by Sharing Strategy", fontsize=14)
ax.set_ylabel("Parameters (Millions)")
ax.set_xlabel("Sharing Strategy")

ax.grid(axis="y", linestyle="--", alpha=0.6)
ax.grid(axis="x", visible=False)

# Add values above bars
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 1,
        f"{height:.1f}",
        ha="center",
        fontsize=9,
    )

plt.tight_layout()

plt.savefig(
    PLOTS / "parameter_count_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()



# ------------------------------------------------------------------
# SST-2 Accuracy
# ------------------------------------------------------------------

df = pd.read_csv(RESULTS)

df = df[df["task_name"] == "sst2"].copy()

df["configuration"] = pd.Categorical(
    df["configuration"],
    categories=order,
    ordered=True,
)

df = df.sort_values("configuration")

plt.style.use("ggplot")

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    [labels[c] for c in df["configuration"]],
    df["accuracy"],
)

ax.set_title("SST-2 Accuracy by Sharing Strategy", fontsize=14)
ax.set_ylabel("Accuracy")
ax.set_xlabel("Sharing Strategy")

ax.set_ylim(0.75, 0.85)

ax.grid(axis="y", linestyle="--", alpha=0.6)
ax.grid(axis="x", visible=False)

for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.001,
        f"{height:.3f}",
        ha="center",
        fontsize=9,
    )

plt.tight_layout()

plt.savefig(
    PLOTS / "sst2_accuracy.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

# ------------------------------------------------------------------
# MRPC Accuracy and F1
# ------------------------------------------------------------------

df = pd.read_csv(RESULTS)

df = df[df["task_name"] == "mrpc"].copy()

order = [
    "no_sharing_mrpc",
    "shared_attention_mrpc",
    "shared_ffn_mrpc",
    "lower_half_sharing_mrpc",
    "upper_half_sharing_mrpc",
    "full_sharing_mrpc",
]

labels = {
    "no_sharing_mrpc": "No Sharing",
    "shared_attention_mrpc": "Shared\nAttention",
    "shared_ffn_mrpc": "Shared\nFFN",
    "lower_half_sharing_mrpc": "Lower\nHalf",
    "upper_half_sharing_mrpc": "Upper\nHalf",
    "full_sharing_mrpc": "Full\nSharing",
}

df["configuration"] = pd.Categorical(
    df["configuration"],
    categories=order,
    ordered=True,
)

df = df.sort_values("configuration")

plt.style.use("ggplot")

fig, ax = plt.subplots(figsize=(9, 5))

x = range(len(df))
width = 0.35

bars1 = ax.bar(
    [i - width/2 for i in x],
    df["accuracy"],
    width,
    label="Accuracy",
)

bars2 = ax.bar(
    [i + width/2 for i in x],
    df["f1"],
    width,
    label="F1 Score",
)

ax.set_title("MRPC Performance by Sharing Strategy", fontsize=14)
ax.set_ylabel("Score")
ax.set_xlabel("Sharing Strategy")

ax.set_xticks(list(x))
ax.set_xticklabels([labels[c] for c in df["configuration"]])

ax.set_ylim(0.60, 0.85)

ax.legend()

ax.grid(axis="y", linestyle="--", alpha=0.6)
ax.grid(axis="x", visible=False)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.001,
            f"{height:.3f}",
            ha="center",
            fontsize=8,
        )

plt.tight_layout()

plt.savefig(
    PLOTS / "mrpc_metrics.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

# ------------------------------------------------------------------
# Parameter Count vs SST-2 Accuracy
# ------------------------------------------------------------------

df = pd.read_csv(RESULTS)

df = df[df["task_name"] == "sst2"].copy()

order = [
    "no_sharing",
    "shared_attention",
    "shared_ffn",
    "lower_half_sharing",
    "upper_half_sharing",
    "full_sharing",
]

labels = {
    "no_sharing": "No Sharing",
    "shared_attention": "Shared Attention",
    "shared_ffn": "Shared FFN",
    "lower_half_sharing": "Lower Half",
    "upper_half_sharing": "Upper Half",
    "full_sharing": "Full Sharing",
}

df["configuration"] = pd.Categorical(
    df["configuration"],
    categories=order,
    ordered=True,
)

df = df.sort_values("configuration")

plt.style.use("ggplot")

fig, ax = plt.subplots(figsize=(8, 6))

x = df["parameters"] / 1_000_000
y = df["accuracy"]

ax.scatter(x, y, s=80)

for _, row in df.iterrows():
    ax.annotate(
        labels[row["configuration"]],
        (row["parameters"] / 1_000_000, row["accuracy"]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=9,
    )

ax.set_title("Parameter Count vs. SST-2 Accuracy", fontsize=14)

ax.set_xlabel("Parameters (Millions)")
ax.set_ylabel("Accuracy")

ax.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()

plt.savefig(
    PLOTS / "parameter_vs_accuracy.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

# ------------------------------------------------------------------
# Training Time (SST-2)
# ------------------------------------------------------------------

df = pd.read_csv(RESULTS)

df = df[df["task_name"] == "sst2"].copy()

order = [
    "no_sharing",
    "shared_attention",
    "shared_ffn",
    "lower_half_sharing",
    "upper_half_sharing",
    "full_sharing",
]

labels = {
    "no_sharing": "No Sharing",
    "shared_attention": "Shared\nAttention",
    "shared_ffn": "Shared\nFFN",
    "lower_half_sharing": "Lower\nHalf",
    "upper_half_sharing": "Upper\nHalf",
    "full_sharing": "Full\nSharing",
}

df["configuration"] = pd.Categorical(
    df["configuration"],
    categories=order,
    ordered=True,
)

df = df.sort_values("configuration")

# Convert seconds to minutes
df["training_time_minutes"] = df["training_time_seconds"] / 60

plt.style.use("ggplot")

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    [labels[c] for c in df["configuration"]],
    df["training_time_minutes"],
)

ax.set_title("SST-2 Training Time by Sharing Strategy", fontsize=14)
ax.set_ylabel("Training Time (Minutes)")
ax.set_xlabel("Sharing Strategy")

ax.grid(axis="y", linestyle="--", alpha=0.6)
ax.grid(axis="x", visible=False)

for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.2,
        f"{height:.1f}",
        ha="center",
        fontsize=9,
    )

plt.tight_layout()

plt.savefig(
    PLOTS / "training_time.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()