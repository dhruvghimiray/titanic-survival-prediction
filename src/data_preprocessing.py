import pandas as pd
from sklearn.preprocessing import LabelEncoder


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
