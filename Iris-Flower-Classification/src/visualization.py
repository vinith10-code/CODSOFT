"""Visualization utilities for Iris exploratory analysis."""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def generate_visualizations(data: pd.DataFrame, images_dir: Path) -> None:
    """Create and save all required visualizations."""
    images_dir.mkdir(parents=True, exist_ok=True)
    plot_data = data.copy()

    if "Id" in plot_data.columns:
        plot_data = plot_data.drop(columns=["Id"])

    sns.set_theme(style="whitegrid")

    _save_pairplot(plot_data, images_dir / "pairplot.png")
    _save_correlation_heatmap(plot_data, images_dir / "correlation_heatmap.png")
    _save_feature_histograms(plot_data, images_dir / "feature_histograms.png")
    _save_class_distribution(plot_data, images_dir / "class_distribution.png")

    print("\nVisualizations saved.")


def _save_pairplot(data: pd.DataFrame, output_path: Path) -> None:
    pairplot = sns.pairplot(data, hue="Species", diag_kind="hist")
    pairplot.fig.suptitle("Iris Feature Pair Plot", y=1.02)
    pairplot.savefig(output_path, bbox_inches="tight")
    plt.close(pairplot.fig)


def _save_correlation_heatmap(data: pd.DataFrame, output_path: Path) -> None:
    numeric_data = data.select_dtypes(include="number")

    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_data.corr(), annot=True, cmap="viridis", linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _save_feature_histograms(data: pd.DataFrame, output_path: Path) -> None:
    numeric_data = data.select_dtypes(include="number")

    numeric_data.hist(figsize=(10, 8), bins=20, color="#4C78A8", edgecolor="black")
    plt.suptitle("Iris Feature Histograms")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _save_class_distribution(data: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=data, x="Species", palette="Set2", hue="Species", legend=False)
    plt.title("Iris Class Distribution")
    plt.xlabel("Species")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
