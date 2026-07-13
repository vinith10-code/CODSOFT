from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

try:
    from .data_loader import load_dataset
    from .preprocessing import preprocess_data
except ImportError:
    from data_loader import load_dataset
    from preprocessing import preprocess_data


MODEL_FILE_NAME = "titanic_model.pkl"


def get_model_path() -> Path:
    """Return the default path for the saved Titanic model."""
    project_root = Path(__file__).resolve().parent.parent
    return project_root / "models" / MODEL_FILE_NAME


def load_processed_data(
    dataset_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load the dataset and return processed features and labels."""
    try:
        data = load_dataset(dataset_path)
        return preprocess_data(data)
    except Exception as error:
        raise RuntimeError("Failed to load and preprocess Titanic data.") from error


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and labels into train and test sets."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def get_models(random_state: int = 42) -> dict[str, ClassifierMixin]:
    """Create the candidate classifiers used for comparison."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree Classifier": DecisionTreeClassifier(
            random_state=random_state
        ),
        "Random Forest Classifier": RandomForestClassifier(
            random_state=random_state
        ),
    }


def train_and_compare_models(
    models: dict[str, ClassifierMixin],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[str, ClassifierMixin, dict[str, float]]:
    """Train each candidate model and compare test accuracy."""
    scores: dict[str, float] = {}
    fitted_models: dict[str, ClassifierMixin] = {}

    try:
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            scores[model_name] = accuracy_score(y_test, predictions)
            fitted_models[model_name] = model
    except Exception as error:
        raise RuntimeError("Failed while training and comparing models.") from error

    best_model_name = max(scores, key=scores.get)
    return best_model_name, fitted_models[best_model_name], scores


def save_model(model: ClassifierMixin, model_path: Path | None = None) -> Path:
    """Save a fitted model and return its path."""
    path = model_path or get_model_path()

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
    except Exception as error:
        raise RuntimeError(f"Failed to save model to {path}.") from error

    return path


def train_model(
    dataset_path: Path | None = None,
    model_path: Path | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[ClassifierMixin, pd.DataFrame, pd.Series, Any]:
    """Train the candidate models and save the best-performing one."""
    features, target = load_processed_data(dataset_path)
    X_train, X_test, y_train, y_test = split_dataset(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )

    models = get_models(random_state=random_state)
    _, best_model, _ = train_and_compare_models(
        models,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    save_model(best_model, model_path)
    predictions = best_model.predict(X_test)

    return best_model, X_test, y_test, predictions


if __name__ == "__main__":
    train_model()
