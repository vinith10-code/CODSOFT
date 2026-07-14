"""Evaluation and persistence utilities for trained Iris models."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_model(
    model: object,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    class_names: list[str],
) -> None:
    """Print model evaluation metrics."""
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    print("\nModel Evaluation")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification Report")
    print(classification_report(y_test, predictions, target_names=class_names))


def save_model(model: object, model_path: Path) -> None:
    """Save the selected model using joblib."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print("\nModel saved successfully.")
