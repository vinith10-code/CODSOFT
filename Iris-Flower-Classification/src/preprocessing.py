"""Preprocessing helpers for the Iris classification workflow."""

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def preprocess_dataset(
    data: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, LabelEncoder, list[str]]:
    """Clean, encode, and split the Iris dataset."""
    processed_data = data.copy()

    if "Id" in processed_data.columns:
        processed_data = processed_data.drop(columns=["Id"])

    if "Species" not in processed_data.columns:
        raise ValueError("The dataset must contain a 'Species' target column.")

    feature_columns = [
        column for column in processed_data.columns if column != "Species"
    ]

    if not feature_columns:
        raise ValueError("The dataset must contain at least one feature column.")

    for column in feature_columns:
        if processed_data[column].isnull().any():
            if pd.api.types.is_numeric_dtype(processed_data[column]):
                processed_data[column] = processed_data[column].fillna(
                    processed_data[column].median()
                )
            else:
                processed_data[column] = processed_data[column].fillna(
                    processed_data[column].mode()[0]
                )

    if processed_data["Species"].isnull().any():
        processed_data["Species"] = processed_data["Species"].fillna(
            processed_data["Species"].mode()[0]
        )

    label_encoder = LabelEncoder()
    processed_data["Species"] = label_encoder.fit_transform(
        processed_data["Species"]
    )

    x = processed_data[feature_columns]
    y = processed_data["Species"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print("\nData preprocessing completed.")
    return x_train, x_test, y_train, y_test, label_encoder, feature_columns
