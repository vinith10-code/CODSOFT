from pathlib import Path

import pandas as pd


DATASET_FILE_NAME = "Titanic-Dataset.csv"


def get_dataset_path() -> Path:
    """Return the default Titanic dataset path."""
    project_root = Path(__file__).resolve().parent.parent
    return project_root / "dataset" / DATASET_FILE_NAME


def validate_dataset_path(dataset_path: Path) -> None:
    """Raise an error if the dataset file is missing."""
    if not dataset_path.exists():
        raise FileNotFoundError(
            "Titanic dataset was not found. Please place "
            f"{DATASET_FILE_NAME} inside the dataset folder: "
            f"{dataset_path.parent}"
        )


def load_dataset(dataset_path: Path | None = None) -> pd.DataFrame:
    """Load the Titanic dataset into a pandas DataFrame."""
    path = dataset_path or get_dataset_path()
    validate_dataset_path(path)

    return pd.read_csv(path)
