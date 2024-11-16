import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Data Preprocessing
def load_data(file_path):
    """ Load data from CSV file """
    return pd.read_csv(file_path)

def clean_data(data):
    """ Handle missing values """
    # Fill missing Age values with median
    data['Age'] = data['Age'].fillna(data['Age'].median())  # No inplace=True
    
    # Drop rows where 'Embarked' is NaN
    data.dropna(subset=['Embarked'], inplace=True)
    
    return data

def feature_engineering(data):
    """ Create and engineer features """
    # Create FamilySize feature
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    
    # Encode 'Sex' column: Female -> 0, Male -> 1
    data['Sex'] = data['Sex'].replace({'female': 0, 'male': 1})
    
    # Extract the first letter from 'Cabin' and fill NaN with 'U' (Unknown)
    data['Cabin_first_letter'] = data['Cabin'].fillna('U').str[0]
    
    # Encode 'Cabin_first_letter' using LabelEncoder
    label_encoder = LabelEncoder()
    data['Cabin_first_letter'] = label_encoder.fit_transform(data['Cabin_first_letter'])
    
    # Encode 'Embarked' column using LabelEncoder
    data['Embarked'] = label_encoder.fit_transform(data['Embarked'].fillna('U'))  # Replace NaN with 'U' for Unknown
    
    return data  

def drop_unnecessary_columns(data, is_training=True):
    """ Drop unnecessary columns and ensure PassengerId is excluded during training """
    columns_to_drop = [
        'Body', 'Hometown', 'Destination', 'Name_wiki', 'WikiId', 
        'Class', 'Boarded', 'Name', 'Ticket', 'Age_wiki', 'Lifeboat', 
        'SibSp', 'Parch', 'Fare', 'Cabin'
    ]
    
    # Drop PassengerId during training but keep it during prediction
    if is_training:
        data.drop(columns=['PassengerId'] + columns_to_drop, inplace=True)
    else:
        data.drop(columns=columns_to_drop, inplace=True)
    
    return data

def preprocess_data(file_path, is_training=True):
    """ Full pipeline to preprocess data """
    data = load_data(file_path)
    data = clean_data(data)
    data = feature_engineering(data)
    data = drop_unnecessary_columns(data, is_training)
    return data


# Model Training and Evaluation
def load_and_preprocess_data(file_path):
    """ Load and preprocess data """
    return preprocess_data(file_path)

def train_model(train_data):
    """ Train the model """
    # Features and target variable
    X = train_data.drop('Survived', axis=1)
    y = train_data['Survived']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define and train the model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    # Make predictions and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy}')
    
    return model, X_train, y_train, X_test, y_test

def tune_model(X_train, y_train):
    """ Hyperparameter tuning with GridSearchCV """
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    }
    
    # Create the model
    model = RandomForestClassifier(random_state=42)
    
    # Perform GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    print(f"Best Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def save_model(model, model_filename):
    """ Save the trained model to a file """
    os.makedirs(os.path.dirname(model_filename), exist_ok=True)
    joblib.dump(model, model_filename)

def load_model(model_filename):
    """ Load a pre-trained model from a file """
    return joblib.load(model_filename)


# Feature Importance Visualization
def plot_feature_importance(model, X_train):
    """ Plot feature importance from RandomForestClassifier """
    feature_importances = model.feature_importances_
    feature_names = X_train.columns
    
    plt.barh(feature_names, feature_importances)
    plt.xlabel('Feature Importance')
    plt.title('Feature Importances from Random Forest')
    plt.show()


# Example of training
train_data = load_and_preprocess_data('data/train.csv')
model, X_train, y_train, X_test, y_test = train_model(train_data)

# Optionally tune the model
best_model = tune_model(X_train, y_train)

# Save the best model
save_model(best_model, 'model/titanic_model.pkl')

# Plot feature importance
plot_feature_importance(best_model, X_train)

