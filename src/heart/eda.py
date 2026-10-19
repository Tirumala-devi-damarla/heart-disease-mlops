"""Exploratory data analysis - saves all plots into reports/figures/."""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from heart import config  # noqa: E402
from heart.data import load_clean, load_raw  # noqa: E402

logger = logging.getLogger(__name__)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# two colours used everywhere: teal = healthy, coral = disease
PALETTE = {0: "#2a9d8f", 1: "#e76f51"}
LABELS = {0: "No disease", 1: "Disease"}

sns.set_theme(style="whitegrid", context="notebook")


def _save(fig, name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_missing_values(raw: pd.DataFrame, out_dir: Path) -> Path:
    counts = raw.isna().sum()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    counts.plot.bar(ax=ax, color="#8d99ae")
    for i, v in enumerate(counts):
        if v:
            ax.text(i, v + 0.1, str(v), ha="center")
    ax.set_title("Missing values per column (raw data)")
    ax.set_ylabel("count")
    return _save(fig, "01_missing_values.png", out_dir)


def plot_class_balance(df: pd.DataFrame, out_dir: Path) -> Path:
    counts = df[config.TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar([LABELS[i] for i in counts.index], counts.values,
                  color=[PALETTE[i] for i in counts.index])
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 2,
                f"{v} ({v / len(df):.0%})", ha="center")
    ax.set_title("Class balance of the target")
    ax.set_ylabel("patients")
    return _save(fig, "02_class_balance.png", out_dir)


def plot_numeric_histograms(df: pd.DataFrame, out_dir: Path) -> Path:
    cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, col in zip(axes.flat, cols):
        for label in (0, 1):
            sns.histplot(df.loc[df[config.TARGET] == label, col], ax=ax,
                         kde=True, color=PALETTE[label], label=LABELS[label],
                         alpha=0.5, bins=20)
        ax.set_title(col)
        ax.legend()
    axes.flat[-1].axis("off")
    fig.suptitle("Distribution of numeric features by target", fontsize=14)
    return _save(fig, "03_numeric_histograms.png", out_dir)


def plot_categorical_counts(df: pd.DataFrame, out_dir: Path) -> Path:
    cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.flat, cols):
        sns.countplot(data=df, x=col, hue=config.TARGET, ax=ax,
                      palette=PALETTE)
        ax.set_title(col)
        ax.legend(title="target", labels=[LABELS[0], LABELS[1]], fontsize=8)
    fig.suptitle("Categorical features vs target", fontsize=14)
    return _save(fig, "04_categorical_counts.png", out_dir)


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: Path) -> Path:
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, linewidths=0.4, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation heatmap (Pearson)")
    return _save(fig, "05_correlation_heatmap.png", out_dir)


def plot_target_correlation(df: pd.DataFrame, out_dir: Path) -> Path:
    corr = (df.corr(numeric_only=True)[config.TARGET]
            .drop(config.TARGET).sort_values())
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [PALETTE[1] if v > 0 else PALETTE[0] for v in corr]
    ax.barh(corr.index, corr.values, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Correlation of each feature with the target")
    return _save(fig, "06_target_correlation.png", out_dir)


def plot_boxplots(df: pd.DataFrame, out_dir: Path) -> Path:
    cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    long = df.melt(id_vars=config.TARGET, value_vars=cols)
    fig, axes = plt.subplots(1, len(cols), figsize=(16, 4))
    for ax, col in zip(axes, cols):
        sns.boxplot(data=long[long.variable == col], x=config.TARGET,
                    y="value", hue=config.TARGET, palette=PALETTE, ax=ax,
                    legend=False)
        ax.set_title(col)
        ax.set_xticks([0, 1], [LABELS[0], LABELS[1]])
        ax.set_xlabel("")
    fig.suptitle("Outlier check - boxplots by target", fontsize=14)
    return _save(fig, "07_boxplots.png", out_dir)


def run_eda(out_dir: Path = config.FIGURES_DIR) -> list:
    raw = load_raw()
    df = load_clean()
    paths = [
        plot_missing_values(raw, out_dir),
        plot_class_balance(df, out_dir),
        plot_numeric_histograms(df, out_dir),
        plot_categorical_counts(df, out_dir),
        plot_correlation_heatmap(df, out_dir),
        plot_target_correlation(df, out_dir),
        plot_boxplots(df, out_dir),
    ]
    summary = df.describe().T.round(2)
    summary.to_csv(out_dir / "summary_statistics.csv")
    logger.info("Saved %d figures to %s", len(paths), out_dir)
    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    run_eda()
