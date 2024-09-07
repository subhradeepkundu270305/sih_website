from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load your pre-trained linear regression model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Example of how to prepare data, replace this with your actual data processing code
def preprocess_data(data):
    # Convert 'is_weekend' to boolean
    data[-1] = data[-1].lower() in ['true', '1', 'yes']
    
    # Convert input data to DataFrame
    df = pd.DataFrame([data], columns=[
        'admin1_encoded', 'market_encoded', 'category_encoded', 
        'commodity_encoded', 'year', 'month', 'day_of_week', 'is_weekend'
    ])
    
    # Add any additional data processing steps here if needed
    return df

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from the incoming request
    data = [request.form.get(key) for key in [
        'admin1_encoded', 'market_encoded', 'category_encoded', 
        'commodity_encoded', 'year', 'month', 'day_of_week', 'is_weekend'
    ]]
    
    # Preprocess the input data
    processed_data = preprocess_data(data)
    
    # Ensure that the processed_data is in the correct format (numpy array or DataFrame)
    features = processed_data.values  # Convert DataFrame to NumPy array if necessary
        
    # Predict using the model
    prediction = model.predict(features)
    
    # Output the prediction
    output = prediction[0]
    
    return render_template('index.html', prediction_text=f'Predicted Price INR: {output}')

if __name__ == "__main__":
    app.run(debug=True)
