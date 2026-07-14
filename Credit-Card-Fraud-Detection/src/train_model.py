from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


MODEL_FILE_NAME = "credit_card_model.pkl"


def get_model_path() -> Path:
    """Return the default path for the saved fraud detection model."""
    project_root = Path(__file__).resolve().parent.parent
    return project_root / "models" / MODEL_FILE_NAME


def split_dataset(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and labels into train and test sets."""
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )


def get_models(random_state: int = 42) -> dict[str, ClassifierMixin]:
    """Create the candidate classifiers used for comparison."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
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

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        scores[model_name] = accuracy_score(y_test, predictions)
        fitted_models[model_name] = model

    best_model_name = max(scores, key=scores.get)
    return best_model_name, fitted_models[best_model_name], scores


def save_model(model: ClassifierMixin, model_path: Path | None = None) -> Path:
    """Save a fitted model and return its path."""
    path = model_path or get_model_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

    return path
