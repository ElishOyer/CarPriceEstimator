# Car Price Prediction Project

## Overview

This project uses machine learning to predict car prices based on various features. It includes scripts for data preparation, model training, and making predictions. Additionally, the project generates a web page that allows users to input car details and receive estimated prices.

## Project Structure

- `create_unique_values.py`: This script extracts unique values from the dataset and saves them to `unique_values.pkl`. **Make sure to run this script before running the machine learning models or other scripts that rely on this data.**
- `car_data_prep.py`: Contains functions for preparing data for model input.
- `model_training.py`: Script for training the machine learning model.
- `predict_price.py`: Script for making predictions using the trained model.
- `app.py`: Flask application that serves the web page and handles user input for price predictions.
- `dataset.csv`: Sample dataset used for training the model.
- `unique_values.pkl`: JSON file with unique values for categorical features (generated by `generate_unique_values.py`).
- `templates/index.html`: HTML form for user input.

## Setup Instructions

1. **Generate Unique Values File**: Run the following script to generate the `unique_values.pkl` file:
   ```bash
   python create_unique_values.py

