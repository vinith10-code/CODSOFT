# Iris Flower Classification

## Project Overview

This project is a complete machine learning workflow for classifying Iris
flowers into species using supervised learning. It loads the dataset, displays
key dataset information, preprocesses the data, creates exploratory
visualizations, trains multiple classification models, selects the best model,
evaluates it, and saves the trained model for reuse.

The project was created for the CodSoft Data Science Internship and is ready for
GitHub upload. The dataset file is intentionally excluded from version control.

## Dataset

Place the Iris dataset file at:

```text
dataset/Iris.csv
```

The dataset must include a `Species` column. If an `Id` column is present, it is
removed automatically during preprocessing.

## Technologies Used

- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib

## Features

- Loads `dataset/Iris.csv`
- Displays rows, columns, column names, missing values, and first five rows
- Removes the `Id` column if present
- Handles missing values
- Encodes species labels
- Splits the dataset into 80% training and 20% testing data
- Generates pair plot, correlation heatmap, feature histograms, and class
  distribution plots
- Trains Logistic Regression, Decision Tree, Random Forest, and K-Nearest
  Neighbors models
- Compares model accuracy and selects the best model automatically
- Displays accuracy, confusion matrix, and classification report
- Saves the best model to `models/iris_model.pkl`

## Project Structure

```text
Iris-Flower-Classification/
|-- dataset/
|   |-- Iris.csv
|   `-- .gitkeep
|-- images/
|   |-- pairplot.png
|   |-- correlation_heatmap.png
|   |-- class_distribution.png
|   |-- feature_histograms.png
|   `-- .gitkeep
|-- models/
|   |-- iris_model.pkl
|   `-- .gitkeep
|-- src/
|   |-- data_loader.py
|   |-- preprocessing.py
|   |-- visualization.py
|   |-- train_model.py
|   `-- evaluate.py
|-- main.py
|-- requirements.txt
|-- README.md
`-- .gitignore
```

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Add `Iris.csv` to the `dataset/` folder, then run:

```bash
python main.py
```

## Results

The program trains and compares four machine learning models:

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors

The model with the highest test accuracy is selected automatically and saved as:

```text
models/iris_model.pkl
```

## Screenshots

After running the project, the following visualizations are generated inside the
`images/` folder:

- `pairplot.png`
- `correlation_heatmap.png`
- `class_distribution.png`
- `feature_histograms.png`

## Future Improvements

- Add hyperparameter tuning with GridSearchCV
- Add cross-validation for more reliable model comparison
- Save the label encoder with the model
- Build a simple web interface for flower species prediction
- Add automated tests for preprocessing and model training
