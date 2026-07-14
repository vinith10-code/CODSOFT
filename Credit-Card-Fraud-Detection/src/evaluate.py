from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate a trained model and print the requested metrics."""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0)

    print("\nModel Evaluation")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix")
    print(matrix)
    print("\nClassification Report")
    print(report)

    return {
        "accuracy": accuracy,
        "confusion_matrix": matrix,
        "classification_report": report,
    }
