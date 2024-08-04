import pandas as pd
import numpy as np
import re

# Summary:
# This code prepares the dataset for further analysis by performing data cleaning and transformation.
# It selects relevant columns, handles missing values, converts data types, cleans the data,
# and calculates the mean price for categorical features.

def prepare_data(df):
    # Select relevant columns
    selected_columns = ['Year', 'Hand', 'Gear', 'capacity_Engine', 'Km', 'manufactor', 'model', 'City', 'Color', 'Price']
    df = df[selected_columns].copy()

    # Replace 'None' values with NaN in all columns
    df.replace('None', np.nan, inplace=True)
    
    # Convert columns to appropriate data types
    df['Gear'] = df['Gear'].astype('category')
    df['capacity_Engine'] = df['capacity_Engine'].astype(str).replace(['None', 'nan'], np.nan).str.replace(',', '').astype(float)
    df['manufactor'] = df['manufactor'].astype('category')
    df['model'] = df['model'].astype('category')
    df['City'] = df['City'].astype('category')
    df['Color'] = df['Color'].astype('category')
    df['Km'] = df['Km'].astype(str).replace(['None', 'nan'], np.nan).str.replace(',', '').astype(float)

    # Fill missing values
    df['Gear'] = df['Gear'].fillna(df['Gear'].mode()[0])
    df['Color'] = df['Color'].fillna(df['Color'].mode()[0])
    df['capacity_Engine'] = df['capacity_Engine'].fillna(df['capacity_Engine'].median())
    df['Km'] = df['Km'].fillna(df['Km'].median())

    # Clean 'model' column by removing the manufacturer name and year
    def clean_model(row):
        model = row['model']
        manufactor = row['manufactor']
        if manufactor in model:
            model = model.replace(manufactor, '').strip()
        if manufactor == 'Lexsus' and 'לקסוס' in model:
            model = model.replace('לקסוס', '').strip()
        model = re.sub(r'\(\d{4}\)', '', model).strip()
        return model

    df['model'] = df.apply(clean_model, axis=1)

    # Calculate mean price for categorical features
    categorical_features = ['manufactor', 'model', 'Gear', 'Color', 'City']
    for feature in categorical_features:
        means = df.groupby(feature)['Price'].mean()
        df[f'{feature}_mean'] = df[feature].map(means)

    # Prepare final DataFrame with selected columns and calculated means
    final_df = df[selected_columns + [f'{feature}_mean' for feature in categorical_features]]

    return final_df
