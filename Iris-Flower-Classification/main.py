"""Main workflow for the Iris Flower Classification project."""

import os
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "Iris.csv"
IMAGES_DIR = BASE_DIR / "images"
MODEL_PATH = BASE_DIR / "models" / "iris_model.pkl"

os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib-cache"))

from src.data_loader import display_dataset_information, load_dataset
from src.evaluate import evaluate_model, save_model
from src.preprocessing import preprocess_dataset
from src.train_model import train_and_select_best_model
from src.visualization import generate_visualizations


def main() -> None:
    """Run the complete Iris classification pipeline."""
    try:
        data = load_dataset(DATASET_PATH)
        display_dataset_information(data)

        x_train, x_test, y_train, y_test, label_encoder, _ = preprocess_dataset(data)

        generate_visualizations(data, IMAGES_DIR)

        best_model, _, _ = train_and_select_best_model(
            x_train,
            x_test,
            y_train,
            y_test,
        )

        evaluate_model(
            best_model,
            x_test,
            y_test,
            class_names=list(label_encoder.classes_),
        )
        save_model(best_model, MODEL_PATH)

        print("\nProject completed successfully.")
    except Exception as exc:
        print(f"\nError: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
