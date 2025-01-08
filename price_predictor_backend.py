
import numpy as np 
import matplotlib.pyplot as pyplot
import pandas as pd
import streamlit as st
import joblib 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import streamlit as st 
import streamlit as st
from datetime import datetime

loaded_preprocessor = joblib.load('preprocessor/preprocessor_predictor.pkl')
   
model_rf = joblib.load('models/new_model_rf.pkl')
model_xgb = joblib.load('models/new_model_xgb.pkl')
model_cat = joblib.load('models/new_model_cat.pkl')


best_weight = [0.32,0.23,0.45]

# Load the preprocessed data    
    #District Mapping
# Define a mapping of towns to districts
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



# Page configuration
st.set_page_config(page_title = "HDB Resale Price Evaluation",layout= 'wide', initial_sidebar_state='expanded')
# Application Header
st.title(" :bar_chart: HDB Resale Price Evaluation")
st.markdown('<style>div.block-container{padding-top:40px; padding-left: 20px;}</style>',unsafe_allow_html=True)
st.write('This application calculates the expected price of your HDB')



st.sidebar.header('HDB Housing Prediction')
st.sidebar.subheader('Please select your HDB details')
# Initialize session state for year and month if not already set
if 'year' not in st.session_state:
    st.session_state['year'] = datetime.now().year
if 'month' not in st.session_state:
    st.session_state['month'] = datetime.now().month

# Sidebar for selecting year and month
st.session_state['year'] = st.sidebar.selectbox(
    'Select year of purchase',
    options=range(2000, datetime.now().year + 1),
    index=(datetime.now().year - 2000)  # Default to current year
)

st.session_state['month'] = st.sidebar.selectbox(
    'Select month of purchase',
    options=range(1, 13),
    format_func=lambda x: datetime(2000, x, 1).strftime('%B'),  # Display month names
    index=(datetime.now().month - 1)  # Default to current month
)
st.session_state['town'] = st.sidebar.selectbox(
    "Select town:",
    options = [    'ANG MO KIO',
    'BEDOK',
    'BISHAN',
    'BUKIT BATOK',
    'BUKIT MERAH',
    'BUKIT TIMAH',
    'CENTRAL AREA',
    'CHOA CHU KANG',
    'CLEMENTI',
    'GEYLANG',
    'HOUGANG',
    'JURONG EAST',
    'JURONG WEST',
    'KALLANG_WHAMPOA',
    'MARINE PARADE',
    'QUEENSTOWN',
    'SENGKANG',
    'SERANGOON',
    'TAMPINES',
    'TOA PAYOH',
    'WOODLANDS',
    'YISHUN',
    'LIM CHU KANG',
    'SEMBAWANG',
    'BUKIT PANJANG',
    'PASIR RIS',
    'PUNGGOL']
)

st.session_state['flat_type'] = st.sidebar.selectbox(
    "Select flat type:",
    options = ['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM']
)

st.session_state['range_numeric'] = st.sidebar.number_input(
    "Enter house floor",
    min_value=1,
    max_value=50,
)
st.session_state['floor_area_sqm'] = st.sidebar.number_input('Enter floor area')
st.session_state['lease_commence_date'] = st.sidebar.selectbox(
    'Enter lease commencement year',
    options=range(1960, datetime.now().year - 5),
    index=(datetime.now().year - 1966)  # Default to current year - 5 due to  MOP
    )

if st.sidebar.button('Evaluate', key = 'Evaluate'):
    if not 'year' in st.session_state or not 'month' in st.session_state or not 'flat_type' in st.session_state or not 'town' in st.session_state or not 'range_numeric' in st.session_state:
        st.sidebar.error('Invalid,Please select all filter')
    else:
        with st.spinner('Evaluating...'):
            st.session_state['new_date'] = int(st.session_state['year']) + int(st.session_state['month'])/12
            st.session_state['district'] = district_mapping.get(st.session_state['town'], 'Unknown')
            new_data = pd.DataFrame({
                'new_date' : st.session_state['new_date'],
                'flat_type': [st.session_state['flat_type']],
                'district': [st.session_state['district']],
                'range_numeric': [st.session_state['range_numeric']],
                'floor_area_sqm': [st.session_state['floor_area_sqm']],    
                'lease_commence_date': [st.session_state['lease_commence_date']]
            })

            st.session_state['prediction'] = get_price(new_data)
#'new_date','flat_type', 'district', 'range_numeric' ,'floor_area_sqm', 'lease_commence_date'
c1,c2 = st.columns((3,7))
with c1:

    st.write("Column 1")
    if 'prediction' in st.session_state:
        prediction = st.session_state['prediction']
        st.write(f"Estimated valuation of home: {prediction}")
      