# Credit Card Fraud Detection

## Project Overview

This project was developed for the CodSoft Data Science Internship. It builds a
simple machine learning pipeline to detect fraudulent credit card transactions
using transaction features from `creditcard.csv`.

The project loads the dataset, displays basic information, performs light
preprocessing, creates visualizations, trains two classification models,
compares their accuracy, evaluates the best model, and saves the trained model.

## Features

- Loads `dataset/creditcard.csv`
- Displays dataset shape, columns, missing values, and class distribution
- Handles missing numeric values with median values
- Scales the `Amount` column
- Drops the `Time` column for a simple baseline model
- Generates and saves exploratory visualizations
- Trains Logistic Regression and Random Forest Classifier
- Compares model accuracy and selects the best model automatically
- Evaluates the best model with accuracy, confusion matrix, and classification report
- Saves the best model as `models/creditcard_model.pkl`

## Project Structure

```text
Credit-Card-Fraud-Detection/
|-- dataset/
|   `-- creditcard.csv
|-- images/
|   `-- Generated visualization files
|-- models/
|   `-- creditcard_model.pkl
|-- src/
|   |-- data_loader.py
|   |-- preprocessing.py
|   |-- visualization.py
|   |-- train_model.py
|   `-- evaluate.py
|-- main.py
|-- README.md
`-- requirements.txt
```

## Dataset

The dataset is not included in this repository because it exceeds GitHub's
100 MB file size limit.

Download the Credit Card Fraud Detection dataset from Kaggle and place the file
inside the `dataset/` folder:

```text
dataset/creditcard.csv
```

The expected target column is `Class`:

- `0` means the transaction is non-fraudulent
- `1` means the transaction is fraudulent

Run the project using:

```bash
python main.py
```

## Models Used

The project trains and compares:

- Logistic Regression
- Random Forest Classifier

The model with the highest test accuracy is selected automatically.

## Visualizations

The following plots are saved in the `images/` folder:

- Fraud vs Non-Fraud Count Plot
- Transaction Amount Histogram
- Confusion Matrix

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

Run the complete workflow:

```bash
python main.py
```

The script will load the dataset, preprocess it, generate visualizations, train
and compare models, evaluate the best model, and save the model file.

## Output

After a successful run, the project creates:

```text
images/fraud_vs_non_fraud_count.png
images/transaction_amount_histogram.png
images/confusion_matrix.png
models/creditcard_model.pkl
```

## Future Improvements

- Add precision, recall, and F1-score focused model selection
- Handle class imbalance with resampling or class weights
- Add cross-validation for more reliable comparison
- Tune model hyperparameters
- Build a small web app for transaction fraud prediction

## Author

Vinith Andrews S

CodSoft Data Science Intern
