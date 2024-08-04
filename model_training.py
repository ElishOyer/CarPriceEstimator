import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from car_data_prep import prepare_data
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Summary:
# This script loads a dataset, prepares it for modeling, splits it into training and testing sets,
# scales the features, trains a Linear Regression model, evaluates the model, and saves both the model and scaler.

# Load the dataset
file_path = r'./dataset.csv'
data = pd.read_csv(file_path)

# Select relevant columns
selected_columns = ['Year', 'Hand', 'Gear', 'capacity_Engine', 'Km', 'manufactor', 'model', 'Price', 'City', 'Color']
df = data[selected_columns].copy()

# Split the data into training and testing sets
train_df, test_df, y_train, y_test = train_test_split(df, df['Price'], test_size=0.2, random_state=42)

# Prepare the data
df_prepared_train = prepare_data(train_df)  # Prepare training data
df_prepared_test = prepare_data(test_df)    # Prepare testing data

# Define features and target for training and testing sets
X_train = df_prepared_train.drop(columns=['Price', 'manufactor', 'model', 'Gear', 'City', 'Color'])
y_train = df_prepared_train['Price']

X_test = df_prepared_test.drop(columns=['Price', 'manufactor', 'model', 'Gear', 'City', 'Color'])
y_test = df_prepared_test['Price']

# Apply StandardScaler to scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, 'models/scaler.pkl')

# Train and evaluate with LinearRegression
linear_reg_model = LinearRegression()
linear_reg_model.fit(X_train_scaled, y_train)

# Predict on test data
predictions_lr = linear_reg_model.predict(X_test_scaled)
predictions_lr = np.maximum(predictions_lr, 0)  # Ensuring non-negative predictions

# Evaluating the model
mse_lr = mean_squared_error(y_test, predictions_lr)
r2_lr = r2_score(y_test, predictions_lr)
rmse_lr = np.sqrt(mse_lr)

# Save the trained model
if not os.path.exists('models'):
    os.makedirs('models')

joblib.dump(linear_reg_model, 'models/linear_reg_model.pkl')

# Summary of the results
print(f'Root Mean Squared Error: {rmse_lr}')
print(f'R2 Score: {r2_lr}')
