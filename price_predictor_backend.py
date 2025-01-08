
import numpy as np 
import matplotlib.pyplot as pyplot
import pandas as pd
import streamlit as st
import joblib 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import streamlit as st 

loaded_preprocessor = joblib.load('preprocessor_predictor.pkl')
   
model_rf = joblib.load('models/new_model_rf.pkl')
model_xgb = joblib.load('models/new_model_xgb.pkl')
model_cat = joblib.load('models/new_model_cat.pkl')


best_weight = [0.32,0.23,0.45]

# Load the preprocessed data    
    #District Mapping
district_mapping = {
    'ANG MO KIO': 'North-East',
    'BEDOK': 'East',
    'BISHAN': 'Central',
    'BUKIT BATOK': 'West',
    'BUKIT MERAH': 'Central',
    'BUKIT TIMAH': 'Central',
    'CENTRAL AREA': 'Central',
    'CHOA CHU KANG': 'West',
    'CLEMENTI': 'West',
    'GEYLANG': 'Central',
    'HOUGANG': 'North-East',
    'JURONG EAST': 'West',
    'JURONG WEST': 'West',
    'KALLANG_WHAMPOA': 'Central',
    'MARINE PARADE': 'East',
    'QUEENSTOWN': 'Central',
    'SENGKANG': 'North-East',
    'SERANGOON': 'North-East',
    'TAMPINES': 'East',
    'TOA PAYOH': 'Central',
    'WOODLANDS': 'North',
    'YISHUN': 'North',
    'LIM CHU KANG': 'West',
    'SEMBAWANG': 'North',
    'BUKIT PANJANG': 'West',
    'PASIR RIS': 'East',
    'PUNGGOL': 'North-East',
}
# Transform an example new data point
new_data = pd.DataFrame({
    'new_date' : 2024,
    'flat_type': ['4 ROOM'],
    'district': ['East'],
    'range_numeric': [12],
    'floor_area_sqm': [105],    
    'lease_commence_date': [1995],
})

def ___init__(self):
    original_data = pd.read_csv('housing_new.csv')
    original_data = original_data.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
    original_data.columns
    original_data = original_data[['new_date','flat_type', 'district', 'range_numeric' ,'floor_area_sqm', 'lease_commence_date']]
    



def prepare_data(data):


    
    return data
def get_price(new_data):

    # Load the preprocessor and transform the new data
    transformed_new_data = loaded_preprocessor.transform(new_data)
    # Extract feature names from the transformers
    categorical_columns = ['district', 'flat_type']
    numerical_columns = ['floor_area_sqm', 'range_numeric']

    num_feature_names = numerical_columns
    cat_feature_names = loaded_preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_columns)
    remainder_feature_names = [col for col in new_data.columns if col not in numerical_columns + categorical_columns]

    # Combine all feature names
    all_feature_names = list(num_feature_names) + list(cat_feature_names) + remainder_feature_names

    # Create a DataFrame with feature names
    transformed_new_data = pd.DataFrame(transformed_new_data, columns=all_feature_names)
    print("Transformed New Data:")
    print(transformed_new_data)


    y_pred = model_xgb.predict(transformed_new_data)*best_weight[0]+ model_rf.predict(transformed_new_data)*best_weight[1] +model_cat.predict(transformed_new_data)*best_weight[2]
    return y_pred







