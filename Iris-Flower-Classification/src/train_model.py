"""Model training and selection for Iris classification."""

from typing import Dict, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def train_and_select_best_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[object, str, Dict[str, float]]:
    """Train multiple classifiers and return the best model by accuracy."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }

    trained_models = {}
    accuracy_scores = {}

    for model_name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        accuracy_scores[model_name] = accuracy_score(y_test, predictions)
        trained_models[model_name] = model

    best_model_name = max(accuracy_scores, key=accuracy_scores.get)
    best_model = trained_models[best_model_name]

    print("\nModel Accuracy Comparison")
    for model_name, accuracy in accuracy_scores.items():
        print(f"{model_name}: {accuracy:.4f}")

    print(f"\nBest model selected: {best_model_name}")
    return best_model, best_model_name, accuracy_scores
