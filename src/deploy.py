from flask import Flask, request, jsonify
import joblib
import pandas as pd
from src.data_preprocessing import preprocess_data

app = Flask(__name__)

# Load model
model = joblib.load('model/titanic_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    """ Predict survival for a given input """
    try:
        # Get the input data from the POST request
        data = request.get_json()
        
        # Preprocess the data
        data_df = pd.DataFrame([data])
        processed_data = preprocess_data(data_df)
        
        # Make prediction
        prediction = model.predict(processed_data)
        
        # Return result as JSON
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
