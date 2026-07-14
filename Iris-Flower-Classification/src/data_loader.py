"""Dataset loading and reporting utilities for Iris classification."""

from pathlib import Path

import pandas as pd


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """Load the Iris dataset from a CSV file."""
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            "Place Iris.csv inside the dataset folder."
        )

    try:
        data = pd.read_csv(dataset_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The dataset file is empty.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError("The dataset file could not be parsed as CSV.") from exc

    if data.empty:
        raise ValueError("The dataset does not contain any rows.")

    print("Dataset loaded successfully.")
    return data


def display_dataset_information(data: pd.DataFrame) -> None:
    """Print key dataset information to the console."""
    print("\nDataset Information")
    print(f"Rows: {data.shape[0]}")
    print(f"Columns: {data.shape[1]}")
    print(f"Column Names: {list(data.columns)}")
    print("\nMissing Values")
    print(data.isnull().sum())
    print("\nFirst Five Rows")
    print(data.head())
