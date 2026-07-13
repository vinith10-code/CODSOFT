from src import data_loader, evaluate, preprocessing, train_model, visualization


def display_dataset_information(data):
    """Print the dataset shape, columns, and missing-value counts."""
    print("\nDataset Information")
    print(f"Rows: {data.shape[0]}")
    print(f"Columns: {data.shape[1]}")
    print("\nColumn Names")
    print(", ".join(data.columns))
    print("\nMissing Values")
    print(data.isna().sum())


def evaluate_best_model(best_model, X_test, y_test):
    """Evaluate the selected model and print classification metrics."""
    return evaluate.evaluate_model(best_model, X_test, y_test)


def run_pipeline():
    """Run the Titanic survival prediction workflow from data to model."""
    print("Starting Titanic Survival Prediction project...\n")

    data = data_loader.load_dataset()
    print("Dataset loaded successfully.")
    display_dataset_information(data)

    features, target = preprocessing.preprocess_data(data)
    print("\nData preprocessing completed.")

    visualization.generate_all_visualizations(data)
    print("Visualizations saved.")

    X_train, X_test, y_train, y_test = train_model.split_dataset(
        features,
        target,
    )
    models = train_model.get_models()
    best_model_name, best_model, model_scores = (
        train_model.train_and_compare_models(
            models,
            X_train,
            y_train,
            X_test,
            y_test,
        )
    )
    print("Model training completed.")

    print("\nModel Accuracy Comparison")
    for model_name, accuracy in model_scores.items():
        print(f"{model_name}: {accuracy:.4f}")

    print(f"\nBest model selected: {best_model_name}")

    evaluate_best_model(best_model, X_test, y_test)
    print("Model evaluation completed.")

    model_path = train_model.save_model(best_model)
    print(f"Model saved successfully: {model_path}")

    print("\nProject completed successfully.")


def main():
    """Execute the project pipeline with clear error reporting."""
    try:
        run_pipeline()
    except Exception as error:
        print("\nAn unexpected error occurred while running the project.")
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
