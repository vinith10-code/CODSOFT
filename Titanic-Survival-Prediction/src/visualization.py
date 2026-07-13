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
    project_root = Path(__file__).resolve().parent.parent
    images_dir = project_root / IMAGE_FOLDER_NAME
    images_dir.mkdir(exist_ok=True)

    return images_dir


def save_plot(file_name: str) -> None:
    """Save the current matplotlib figure and close it."""
    output_path = get_images_dir() / file_name
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def add_legend_if_available(axis, title: str, labels: list[str] | None = None) -> None:
    """Add a legend when the plot exposes legend handles."""
    handles, detected_labels = axis.get_legend_handles_labels()

    if not handles:
        return

    axis.legend(title=title, labels=labels or detected_labels)


def plot_survival_count(data: pd.DataFrame) -> None:
    """Create and save a survival count plot."""
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(data=data, x="Survived", hue="Survived", palette="Set2")
    ax.set_title("Survival Count")
    ax.set_xlabel("Survived")
    ax.set_ylabel("Number of Passengers")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Did Not Survive", "Survived"])
    add_legend_if_available(ax, "Survived", ["No", "Yes"])

    save_plot("survival_count_plot.png")


def plot_passenger_class_distribution(data: pd.DataFrame) -> None:
    """Create and save a passenger class distribution plot."""
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(data=data, x="Pclass", hue="Pclass", palette="Set1")
    ax.set_title("Passenger Class Distribution")
    ax.set_xlabel("Passenger Class")
    ax.set_ylabel("Number of Passengers")
    add_legend_if_available(ax, "Class")

    save_plot("passenger_class_distribution.png")


def plot_gender_distribution(data: pd.DataFrame) -> None:
    """Create and save a gender distribution plot."""
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(data=data, x="Sex", hue="Sex", palette="pastel")
    ax.set_title("Gender Distribution")
    ax.set_xlabel("Gender")
    ax.set_ylabel("Number of Passengers")
    add_legend_if_available(ax, "Gender")

    save_plot("gender_distribution.png")


def plot_age_distribution(data: pd.DataFrame) -> None:
    """Create and save an age distribution histogram."""
    plt.figure(figsize=(8, 6))
    ax = sns.histplot(data=data, x="Age", bins=30, kde=True, color="steelblue")
    ax.set_title("Age Distribution")
    ax.set_xlabel("Age")
    ax.set_ylabel("Number of Passengers")

    save_plot("age_distribution_histogram.png")


def plot_correlation_heatmap(data: pd.DataFrame) -> None:
    """Create and save a correlation heatmap for numeric columns."""
    numeric_data = data.select_dtypes(include=["number"])
    correlation_matrix = numeric_data.corr()

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
    )
    ax.set_title("Correlation Heatmap")

    save_plot("correlation_heatmap.png")


def generate_all_visualizations(data: pd.DataFrame) -> None:
    """Generate and save all Titanic dataset visualizations."""
    plot_survival_count(data)
    plot_passenger_class_distribution(data)
    plot_gender_distribution(data)
    plot_age_distribution(data)
    plot_correlation_heatmap(data)
