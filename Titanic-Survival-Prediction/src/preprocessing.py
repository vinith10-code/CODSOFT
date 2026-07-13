import pandas as pd
from sklearn.preprocessing import OneHotEncoder


TARGET_COLUMN = "Survived"
CABIN_COLUMN = "Cabin"
UNNECESSARY_COLUMNS = ["PassengerId", "Name", "Ticket"]
CATEGORICAL_COLUMNS = ["Sex", "Embarked"]


def validate_data(data: pd.DataFrame) -> None:
    """Ensure the dataset contains the target column."""
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Required target column '{TARGET_COLUMN}' is missing.")


def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """Fill useful missing values and drop the sparse Cabin column."""
    processed_data = data.copy()

    if "Age" in processed_data.columns:
        processed_data["Age"] = processed_data["Age"].fillna(
            processed_data["Age"].median()
        )

    if "Embarked" in processed_data.columns:
        embarked_mode = processed_data["Embarked"].mode()
        if not embarked_mode.empty:
            processed_data["Embarked"] = processed_data["Embarked"].fillna(
                embarked_mode[0]
            )

    if CABIN_COLUMN in processed_data.columns:
        processed_data = processed_data.drop(columns=[CABIN_COLUMN])

    return processed_data


def remove_unnecessary_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Drop identifier and text columns that are not used for training."""
    columns_to_drop = [
        column for column in UNNECESSARY_COLUMNS if column in data.columns
    ]
    return data.drop(columns=columns_to_drop)


def encode_categorical_features(data: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns used by the model."""
    processed_data = data.copy()
    columns_to_encode = [
        column for column in CATEGORICAL_COLUMNS if column in processed_data.columns
    ]

    if not columns_to_encode:
        return processed_data

    encoder = OneHotEncoder(drop="first", sparse_output=False)
    encoded_values = encoder.fit_transform(processed_data[columns_to_encode])
    encoded_columns = encoder.get_feature_names_out(columns_to_encode)

    encoded_data = pd.DataFrame(
        encoded_values,
        columns=encoded_columns,
        index=processed_data.index,
    )

    processed_data = processed_data.drop(columns=columns_to_encode)
    processed_data = pd.concat([processed_data, encoded_data], axis=1)

    return processed_data


def preprocess_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return model-ready features and target labels."""
    validate_data(data)

    processed_data = handle_missing_values(data)
    processed_data = remove_unnecessary_columns(processed_data)
    processed_data = encode_categorical_features(processed_data)

    target = processed_data[TARGET_COLUMN]
    features = processed_data.drop(columns=[TARGET_COLUMN])

    return features, target
