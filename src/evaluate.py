import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from src.data_preprocessing import preprocess_data

def save_predictions_to_csv(predictions, test_data, filename="predictions.csv"):
    """ Save predictions to a CSV file including all columns from test_data """
    # Add the 'Survived' column with predictions
    test_data['Survived'] = predictions
    
    # Save all columns to CSV (excluding PassengerId for training)
    test_data.to_csv(filename, index=False)
    print(f"Predictions saved to {filename}")

def evaluate_model(model_filename, test_data):
    """ Evaluate the model """
    # Load the model
    model = joblib.load(model_filename)
    
    # Ensure PassengerId is dropped for prediction (to match training data)
    X_test = test_data.drop('PassengerId', axis=1, errors='ignore')  # Drop PassengerId if present
    y_test = test_data['Survived'] if 'Survived' in test_data.columns else None
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # If we have actual 'Survived' values, evaluate accuracy
    if y_test is not None:
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Model Accuracy: {accuracy}')
        print('Confusion Matrix:')
        print(confusion_matrix(y_test, y_pred))
        print('Classification Report:')
        print(classification_report(y_test, y_pred))
    else:
        print("Predictions made (no actual 'Survived' values in test data):")
        print(y_pred)
    
    # Save predictions to CSV with PassengerId and other columns
    save_predictions_to_csv(y_pred, test_data)

# Example usage
test_data = preprocess_data('data/test.csv', is_training=False)  # Pass is_training=False for prediction
evaluate_model('model/titanic_model.pkl', test_data)
