import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "images"
MPL_CONFIG_DIR = PROJECT_ROOT / ".matplotlib-cache"

IMAGES_DIR.mkdir(exist_ok=True)
MPL_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
)


CONFUSION_MATRIX_FILE_NAME = "confusion_matrix.png"


def save_confusion_matrix_plot(matrix) -> Path:
    """Save the confusion matrix as a heatmap image."""
    output_path = IMAGES_DIR / CONFUSION_MATRIX_FILE_NAME

    plt.figure(figsize=(6, 5))
    axis = sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Did Not Survive", "Survived"],
        yticklabels=["Did Not Survive", "Survived"],
    )
    axis.set_title("Confusion Matrix")
    axis.set_xlabel("Predicted Label")
    axis.set_ylabel("Actual Label")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate a trained model and save its confusion matrix plot."""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions)

    save_confusion_matrix_plot(matrix)

    print("\nModel Evaluation")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix")
    print(matrix)
    print("\nClassification Report")
    print(report)

    return {
        "accuracy": accuracy,
        "confusion_matrix": matrix,
        "classification_report": report,
    }
