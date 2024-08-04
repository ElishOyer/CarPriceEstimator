import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template, Response
import joblib
import json

# Summary:
# This script sets up a Flask application for car price prediction. It initializes the Flask app, defines routes for 
# the home page and prediction endpoint, and includes functions to prepare data, perform predictions, and validate 
# input data. It uses pre-trained models and scalers stored in files and handles errors during prediction. 
# The app is configured to run on port 5000.

# Initialize the Flask application
app = Flask(__name__)

# Paths to the model, columns, category means, scaler, and unique values
model_path = 'models/trained_model.pkl'
columns_path = 'models/columns.pkl'
means_path = 'models/category_means.pkl'
scaler_path = 'models/scaler.pkl'
unique_values_path = 'models/unique_values.pkl'

# Load unique values
unique_values = joblib.load(unique_values_path)

# Route for the home page
@app.route('/', methods=['GET'])
def home():
    # Render the HTML template 'index.html'
    return render_template('index.html', unique_values=unique_values)

# Route for serving unique values as JSON
@app.route('/unique_values.json')
def get_unique_values():
    unique_values = joblib.load('models/unique_values.pkl')
    return Response(json.dumps(unique_values, ensure_ascii=False), mimetype='application/json')

# Function to prepare the data for prediction
def prepare_data(data, category_means):
    # Default values for missing numerical fields
    default_values = {
        'Km': 50000.0,
        'capacity_Engine': 1600.0
    }

    # Prepare data with default values and category means
    # If a value is missing for a categorical field, it is set to the mean value of that category.
    prepared_data = {
        'Year': int(data['Year']),
        'Hand': int(data['Hand']),
        'Km': float(data['Km']) if data.get('Km') else default_values['Km'],
        'capacity_Engine': float(data['capacity_Engine']) if data.get('capacity_Engine') else default_values['capacity_Engine'],
        'manufactor_mean': category_means['manufactor_mean'].get(data['manufactor'], sum(category_means['manufactor_mean'].values()) / len(category_means['manufactor_mean'])),
        'Gear_mean': category_means['Gear_mean'].get(data['Gear'], sum(category_means['Gear_mean'].values()) / len(category_means['Gear_mean'])),
        'City_mean': category_means['City_mean'].get(data['City'], sum(category_means['City_mean'].values()) / len(category_means['City_mean'])),
        'model_mean': category_means['model_mean'].get(data['model'], sum(category_means['model_mean'].values()) / len(category_means['model_mean'])),
        'Color_mean': category_means['Color_mean'].get(data['Color'], sum(category_means['Color_mean'].values()) / len(category_means['Color_mean'])),
    }

    return prepared_data

# Function to perform prediction
def predict(data):
    try:
        # Load the model, columns, category means, and scaler
        model = joblib.load(model_path)
        columns = joblib.load(columns_path)
        category_means = joblib.load(means_path)
        scaler = joblib.load(scaler_path)

        print("Loaded model:", model)
        print("Model parameters:", model.get_params())

        # Prepare the input data for prediction
        print("category_means :{}", category_means)
        prepared_data = prepare_data(data, category_means)
        print("Prepared data for prediction:", prepared_data)
        
        features = pd.DataFrame([prepared_data], columns=columns)
        print("Features before scaling:", features)
        
        # Standardize the input features
        features_scaled = scaler.transform(features)
        print("Scaled features:", features_scaled)
        
        # Predict using the loaded model
        prediction = model.predict(features_scaled)
        # Convert negative predictions to positive
        # ElasticNet sometimes produces negative values, so we take the absolute value
        prediction = np.abs(prediction)
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': "Error With Predict"}), 500

# Route to handle POST requests for prediction
@app.route('/predict', methods=['POST'])
def index():
    data = request.get_json(force=True)
    print("data :{}", data)
    errors = {}

    # Validate the input data
    required_fields = {
        'manufactor': 'Manufactor is required',
        'Year': 'Year is required',
        'model': 'Model is required',
        'Hand': 'Hand is required',
        'City': 'City is required',
        'Cre_date': 'Creation Date is required'
    }

    errors = {field: message for field, message in required_fields.items() if not data.get(field)}

    # Validate the specific input data
    Year = data.get('Year')
    if Year:
        Year = int(Year)
        if Year < 1900 or Year > 2023:
            errors['Year'] = 'Please enter a valid year (1900 - current year)'
    Hand = data.get('Hand')
    if Hand and Hand not in ['1', '2', '3', '4', '5']:
        errors['Hand'] = 'Please enter a number between 1-5'
    capacity_Engine = data.get('capacity_Engine')
    if capacity_Engine:
        capacity_Engine = float(capacity_Engine)
        if capacity_Engine <= 0 or capacity_Engine > 10000:
            errors['capacity_Engine'] = 'Please enter a valid engine capacity (0 - 10000)'
    Km = data.get('Km')
    if Km:
        Km = float(Km)
        if Km <= 0 or Km > 1000000:
            errors['Km'] = 'Please enter a valid kilometer value (0 - 1000000)'
    manufactor = data.get('manufactor')
    if manufactor and (len(manufactor) < 2 or len(manufactor) > 50):
        errors['manufactor'] = 'Please enter a valid manufactor name (2-50 characters)'
    model = data.get('model')
    if model and (len(model) < 2 or len(model) > 50):
        errors['model'] = 'Please enter a valid model name (2-50 characters)'
    Gear = data.get('Gear')
    if Gear and (len(Gear) < 2 or len(Gear) > 50):
        errors['Gear'] = 'Please enter a valid gear name (2-50 characters)'
    City = data.get('City')
    if City and (len(City) < 2 or len(City) > 50):
        errors['City'] = 'Please enter a valid city name (2-50 characters)'
    Color = data.get('Color')
    if Color and (len(Color) < 2 or len(Color) > 20):
        errors['Color'] = 'Please enter a valid color (2-20 characters)'

    # If there are errors, return them to the client
    if errors:
        return jsonify({'error': errors}), 500

    # Default values for 'Km' and 'capacity_Engine' that represent reasonable averages for a typical car
    if not data.get('Km'):
        data['Km'] = '50000'
    if not data.get('capacity_Engine'):
        data['capacity_Engine'] = '1600'

    # Call the predict function to get the prediction
    predict_data = predict(data)
    return predict_data

# Function to launch the application
def run_flask_app():
    app.run(port=5000)

# Run the Flask application if this script is executed directly
if __name__ == '__main__':
    run_flask_app()
