from src import data_loader, evaluate, preprocessing, train_model, visualization


def display_dataset_information(data):
    """Print the requested dataset overview."""
    print("\nDataset Loaded Successfully")
    print(f"Number of rows: {data.shape[0]}")
    print(f"Number of columns: {data.shape[1]}")

    print("\nColumn Names")
    print(list(data.columns))

    print("\nDataset Information")
    data.info()

    print("\nMissing Values")
    print(data.isna().sum())

    print("\nClass Distribution")
    if "Class" in data.columns:
        print(data["Class"].value_counts().sort_index())
    else:
        print("Class column is missing.")


def run_pipeline():
    """Run the credit card fraud detection workflow from data to model."""
    print("Starting Credit Card Fraud Detection project...\n")

    data = data_loader.load_dataset()
    print("Dataset Loaded...")
    display_dataset_information(data)

    print("\nPreprocessing...")
    features, target = preprocessing.preprocess_data(data)

    visualization.generate_all_visualizations(data)
    print("Visualizations Generated...")

    X_train, X_test, y_train, y_test = train_model.split_dataset(
        features,
        target,
    )

    print("Training Models...")
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

    print("\nModel Accuracy Comparison")
    for model_name, accuracy in model_scores.items():
        print(f"{model_name}: {accuracy:.4f}")

    print(f"\nBest model selected: {best_model_name}")

    print("\nEvaluating Models...")
    evaluate.evaluate_model(best_model, X_test, y_test)

    print("Saving Best Model...")
    model_path = train_model.save_model(best_model)
    print(f"Model saved successfully: {model_path}")

    print("\nProject completed successfully.")


def main():
    """Execute the project pipeline with clear error reporting."""
    try:
        run_pipeline()
    except FileNotFoundError as error:
        print("\nDataset loading failed.")
        print(error)
    except Exception as error:
        print("\nAn unexpected error occurred while running the project.")
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
