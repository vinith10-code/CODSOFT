from pathlib import Path

import pandas as pd


DATASET_FILE_NAME = "creditcard.csv"
MAX_ROWS = 50000
RANDOM_STATE = 42


def get_dataset_path() -> Path:
    """Return the default path to the credit card fraud dataset."""
    project_root = Path(__file__).resolve().parent.parent
    return project_root / "dataset" / DATASET_FILE_NAME


def validate_dataset_path(dataset_path: Path) -> None:
    """Raise an error if the dataset file is missing."""
    if not dataset_path.exists():
        raise FileNotFoundError(
            "Dataset file not found.\n"
            f"Expected path: {dataset_path}\n"
            f"Please place {DATASET_FILE_NAME} inside the dataset folder."
        )


def load_dataset(dataset_path: Path | None = None) -> pd.DataFrame:
    """Load the credit card fraud dataset into a pandas DataFrame."""
    path = dataset_path or get_dataset_path()
    validate_dataset_path(path)

    data = pd.read_csv(path)

    if data.shape[0] > MAX_ROWS:
        data = data.sample(n=MAX_ROWS, random_state=RANDOM_STATE)

    return data.reset_index(drop=True)
