# Titanic Survival Prediction

## Project Overview

This project was developed as part of the **CodSoft Data Science
Internship**. The objective is to build a machine learning pipeline that
predicts whether a Titanic passenger survived based on passenger details such
as age, gender, ticket class, fare, and embarkation port.

The project includes data loading, preprocessing, exploratory visualization,
model training, model comparison, evaluation, and model persistence.

## Features

- Loads the Titanic dataset from `dataset/Titanic-Dataset.csv`
- Displays basic dataset information and missing value counts
- Handles missing values and removes unnecessary columns
- Encodes categorical features for machine learning
- Generates and saves exploratory data visualizations
- Trains multiple classification models
- Compares models using accuracy
- Automatically selects the best-performing model
- Saves the trained model as `models/titanic_model.pkl`
- Provides a reusable, modular project structure

## Project Structure

```text
Titanic-Survival-Prediction/
|-- dataset/
|   `-- Titanic-Dataset.csv
|-- images/
|   `-- Generated visualization files
|-- models/
|   `-- titanic_model.pkl
|-- src/
|   |-- data_loader.py
|   |-- evaluate.py
|   |-- preprocessing.py
|   |-- train_model.py
|   `-- visualization.py
|-- main.py
|-- README.md
`-- requirements.txt
```

## Technologies Used

- Python
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- joblib

## Dataset Information

The dataset used in this project is the Titanic passenger dataset. It contains
passenger-level information such as:

- Passenger ID
- Survival status
- Passenger class
- Name
- Gender
- Age
- Number of siblings or spouses aboard
- Number of parents or children aboard
- Ticket number
- Fare
- Cabin
- Embarkation port

The target column is `Survived`, where:

- `0` means the passenger did not survive
- `1` means the passenger survived

## Machine Learning Models

The project trains and compares the following classification models:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

The best-performing model is selected automatically based on testing accuracy.

## Data Preprocessing

The preprocessing stage prepares the dataset for model training by:

- Validating that the target column exists
- Filling missing `Age` values with the median age
- Filling missing `Embarked` values with the mode
- Dropping the `Cabin` column because of high missing values
- Removing non-essential columns such as `PassengerId`, `Name`, and `Ticket`
- Encoding categorical columns such as `Sex` and `Embarked`
- Splitting the dataset into features and labels

## Data Visualization

The visualization module generates and saves the following plots:

- Survival count plot
- Passenger class distribution plot
- Gender distribution plot
- Age distribution histogram
- Correlation heatmap

All generated plots are saved in the `images/` folder.

## Model Evaluation

The best model is evaluated using:

- Accuracy score
- Confusion matrix
- Classification report

These metrics help measure how well the selected model predicts passenger
survival on unseen test data.

## Installation

Clone the repository and navigate to the project folder:

```bash
git clone <repository-url>
cd Titanic-Survival-Prediction
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Place the Titanic dataset file inside the `dataset/` folder:

```text
dataset/Titanic-Dataset.csv
```

## How to Run

Run the complete machine learning pipeline:

```bash
python main.py
```

The script will load the dataset, preprocess the data, generate visualizations,
train and compare models, evaluate the best model, and save the trained model.

## Sample Output

```text
Starting Titanic Survival Prediction project...

Dataset loaded successfully.
Data preprocessing completed.
Visualizations saved.
Model training completed.

Model Accuracy Comparison
Logistic Regression: 0.8101
Decision Tree Classifier: 0.7765
Random Forest Classifier: 0.8268

Best model selected: Random Forest Classifier
Model evaluation completed.
Model saved successfully: models/titanic_model.pkl

Project completed successfully.
```

## Future Improvements

- Add hyperparameter tuning with GridSearchCV or RandomizedSearchCV
- Include cross-validation for more reliable model comparison
- Add feature scaling for models that benefit from normalized data
- Build a web interface for passenger survival prediction
- Add automated tests for preprocessing and training functions
- Track model metrics and experiment history

## Author

**Vinith Andrews S**

CodSoft Data Science Intern
