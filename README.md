# Titanic Survival Prediction

This project implements a machine learning model to predict the survival of passengers aboard the Titanic based on various features such as age, sex, class, and more. The model is built using a **RandomForestClassifier** and leverages data preprocessing, feature engineering, hyperparameter tuning, and cross-validation to achieve optimal performance.

## Project Structure

```
├── data/                   # Data folder containing the 'train.csv' and 'test.csv'
├── src/                    # Source code for the project
│   ├── data_preprocessing.py   # Data cleaning, feature engineering, and preprocessing functions
│   ├── model.py              # Model training, hyperparameter tuning, and evaluation
│   ├── evaluate.py           # Evaluation script for making predictions and saving results
│   └── utils.py              # Utility functions (optional, if any)
├── model/                   # Directory to store the trained model file
│   └── titanic_model.pkl     # Trained RandomForest model
├── requirements.txt         # List of dependencies
└── README.md                # Project documentation
```

## Getting Started

### Prerequisites

You need Python 3.x and the following libraries to run this project:

- `pandas`
- `scikit-learn`
- `matplotlib`
- `joblib`

You can install these dependencies using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### Dataset

The dataset used for training and testing is based on the Titanic dataset provided by [Kaggle](https://www.kaggle.com/c/titanic/data). It consists of the following files:

- `train.csv`: The training dataset containing passenger information and the survival status (1 = survived, 0 = did not survive).
- `test.csv`: The test dataset for which we need to predict the survival status of passengers.

### Running the Code

#### 1. Data Preprocessing and Model Training

To preprocess the data and train the model, run the following command:

```bash
python -m src.model
```

This will:
- Load the `train.csv` dataset.
- Perform data cleaning, feature engineering, and preprocessing.
- Train the model using `RandomForestClassifier`.
- Perform hyperparameter tuning using `GridSearchCV` to optimize the model's performance.
- Evaluate the model's performance on a test set and output the accuracy score.

#### 2. Evaluating and Making Predictions

To make predictions using the trained model on the test dataset (`test.csv`), run the following command:

```bash
python -m src.evaluate
```

This will:
- Load the `test.csv` dataset.
- Preprocess the data and make predictions on whether each passenger survived.
- Save the predictions to a file named `predictions.csv` which includes all original columns and the predicted `Survived` values.

### Example of `predictions.csv` Output

The saved `predictions.csv` file will include the following columns:
- `PassengerId`: The unique ID of each passenger.
- `Survived`: The predicted survival status (1 = survived, 0 = did not survive).
- All original columns from the test data, except for the dropped features.

## Hyperparameter Tuning

The project uses **GridSearchCV** to perform hyperparameter tuning on the `RandomForestClassifier`. The following hyperparameters are tuned:

- `n_estimators`: Number of trees in the forest.
- `max_depth`: The maximum depth of the tree.
- `min_samples_split`: The minimum number of samples required to split an internal node.
- `min_samples_leaf`: The minimum number of samples required to be at a leaf node.
- `max_features`: The number of features to consider for the best split.

The best hyperparameters found during the search are:
```python
{
    'max_depth': None,
    'max_features': 'sqrt',
    'min_samples_leaf': 4,
    'min_samples_split': 2,
    'n_estimators': 300
}
```

## Feature Importance

After training the model, the **feature importance** is visualized to understand which features contribute most to the prediction. This helps identify which passenger characteristics (such as `Age`, `Sex`, `FamilySize`, etc.) have the greatest impact on survival predictions.

## Dependencies

To run this project, make sure to install the required dependencies using the following:

```bash
pip install -r requirements.txt
```

Here is a sample of the dependencies that are required for this project:
- `pandas==1.3.3`
- `scikit-learn==0.24.2`
- `matplotlib==3.4.3`
- `joblib==1.0.1`

## Conclusion

This project is a simple yet powerful application of machine learning for classification tasks. By leveraging the Titanic dataset, we have demonstrated data preprocessing, feature engineering, model training, and hyperparameter tuning to build an accurate survival prediction model.

---
