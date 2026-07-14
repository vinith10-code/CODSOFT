import pandas as pd
from sklearn.preprocessing import StandardScaler


TARGET_COLUMN = "Class"
AMOUNT_COLUMN = "Amount"
TIME_COLUMN = "Time"


def validate_data(data: pd.DataFrame) -> None:
    """Ensure the dataset contains the target column."""
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Required target column '{TARGET_COLUMN}' is missing.")


def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """Fill numeric missing values with each column median."""
    processed_data = data.copy()
    numeric_columns = processed_data.select_dtypes(include=["number"]).columns

    for column in numeric_columns:
        if processed_data[column].isna().any():
            processed_data[column] = processed_data[column].fillna(
                processed_data[column].median()
            )

    return processed_data


def scale_amount_column(data: pd.DataFrame) -> pd.DataFrame:
    """Scale the Amount column when it is available."""
    processed_data = data.copy()

    if AMOUNT_COLUMN in processed_data.columns:
        scaler = StandardScaler()
        processed_data[AMOUNT_COLUMN] = scaler.fit_transform(
            processed_data[[AMOUNT_COLUMN]]
        )

    return processed_data


def remove_time_column(data: pd.DataFrame) -> pd.DataFrame:
    """Drop the Time column because it is not needed for this baseline model."""
    if TIME_COLUMN not in data.columns:
        return data

    return data.drop(columns=[TIME_COLUMN])


def preprocess_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return model-ready features and target labels."""
    validate_data(data)

    processed_data = handle_missing_values(data)
    processed_data = scale_amount_column(processed_data)
    processed_data = remove_time_column(processed_data)

    target = processed_data[TARGET_COLUMN]
    features = processed_data.drop(columns=[TARGET_COLUMN])

    return features, target
