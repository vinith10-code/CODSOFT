import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MPL_CONFIG_DIR = PROJECT_ROOT / ".matplotlib-cache"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402


IMAGE_FOLDER_NAME = "images"


def get_images_dir() -> Path:
    """Return the project images directory, creating it if needed."""
    images_dir = PROJECT_ROOT / IMAGE_FOLDER_NAME
    images_dir.mkdir(exist_ok=True)

    return images_dir


def save_plot(file_name: str) -> None:
    """Save the current matplotlib figure and close it."""
    output_path = get_images_dir() / file_name
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_class_distribution(data: pd.DataFrame) -> None:
    """Create and save the fraud vs non-fraud count plot."""
    plt.figure(figsize=(8, 6))
    axis = sns.countplot(data=data, x="Class", hue="Class", palette="Set2")
    axis.set_title("Fraud vs Non-Fraud Transactions")
    axis.set_xlabel("Transaction Class")
    axis.set_ylabel("Number of Transactions")
    axis.set_xticks([0, 1])
    axis.set_xticklabels(["Non-Fraud", "Fraud"])
    axis.legend(title="Class", labels=["Non-Fraud", "Fraud"])

    save_plot("fraud_vs_non_fraud_count.png")


def plot_amount_distribution(data: pd.DataFrame) -> None:
    """Create and save the transaction amount histogram."""
    plt.figure(figsize=(8, 6))
    axis = sns.histplot(data=data, x="Amount", bins=50, kde=True, color="steelblue")
    axis.set_title("Transaction Amount Distribution")
    axis.set_xlabel("Transaction Amount")
    axis.set_ylabel("Number of Transactions")

    save_plot("transaction_amount_histogram.png")


def plot_correlation_heatmap(data: pd.DataFrame) -> None:
    """Create and save a correlation heatmap for numeric columns."""
    numeric_data = data.select_dtypes(include=["number"])
    correlation_matrix = numeric_data.corr()

    plt.figure(figsize=(14, 10))
    axis = sns.heatmap(
        correlation_matrix,
        cmap="coolwarm",
        linewidths=0.2,
        linecolor="white",
    )
    axis.set_title("Correlation Heatmap")

    save_plot("correlation_heatmap.png")


def generate_all_visualizations(data: pd.DataFrame) -> None:
    """Generate and save all project visualizations."""
    plot_class_distribution(data)
    plot_amount_distribution(data)
    plot_correlation_heatmap(data)
