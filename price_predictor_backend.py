
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
import seaborn as sns
import matplotlib.pyplot as plt

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


def create_card(label, value):
    return f"""
        <div style="
            border:2px solid #FFFFFF;
            border-radius:5px;
            padding:5px;
            margin: 3px 0px;
            background-color:#333333;
            color:white;
            width: 400px;  /* Fixed width for smaller card */
            font-size: 80px;  /* Smaller font size */
        ">
            <h4 style="margin: 0px; font-weight: bold; color:white; font-size: 22px;">{label}</h4>
            <h2 style="margin: 0px; color:white; font-size: 22px;">{value}</h2>
        </div>
    """


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


    y_pred = model_xgb.predict(transformed_new_data)*best_weight[0]+ model_rf.predict(transformed_new_data)*best_weight[1] +model_cat.predict(transformed_new_data)*best_weight[2]
    return y_pred
def calculate_hdb_loan_details(purchase_price, tenure_years, grants ):
    # Constants
    interest_rate=2.6
    ltv=80
    stamp_duty_rate=3
    monthly_interest_rate = (interest_rate / 100) / 12
    loan_amount = float(purchase_price) * 0.8 - grants
    num_payments = tenure_years * 12

    # Check if loan amount is valid
    if loan_amount <= 0:
        raise ValueError("Loan amount is zero or negative. Check the CPF grants or purchase price.")

    # Buyer's Stamp Duty (BSD)
    bsd = calculate_stamp_duty(purchase_price, stamp_duty_rate)

    # Monthly payment (Amortization formula)
    monthly_payment = loan_amount * (monthly_interest_rate * (1 + monthly_interest_rate)**num_payments) \
                      / ((1 + monthly_interest_rate)**num_payments - 1)

    # Monthly breakdown of principal and interest
    breakdown = []
    remaining_balance = loan_amount
    for _ in range(num_payments):
        interest_payment = remaining_balance * monthly_interest_rate
        principal_payment = monthly_payment - interest_payment
        remaining_balance -= principal_payment
        breakdown.append((principal_payment, interest_payment, remaining_balance))

    # Format breakdown to ensure it's easier to use in displays
    breakdown = [{"Principal": round(float(p),2), "Interest": round(float(i),2), "Remaining Balance": round(float(r),2)} 
                 for p, i, r in breakdown]
    # Format numbers to 2 significant figures

    return {
        "purchase_price": round((purchase_price.item()),2),
        "loan_amount": round((loan_amount.item()),2),
        "tenure_years": round((tenure_years.item()),2),
        "monthly_payment": round((monthly_payment.item()),2),
        "bsd": round((bsd.item()),2),
        "breakdown": breakdown
    }

def calculate_stamp_duty(price, rate):
    if price <= 180000:
        return price * 0.01
    elif price <= 360000:
        return (180000 * 0.01) + ((price - 180000) * 0.02)
    else:
        return (180000 * 0.01) + (180000 * 0.02) + ((price - 360000) * rate / 100)

# Page configuration
st.set_page_config(page_title = "HDB Resale Price Evaluation",layout= 'wide', initial_sidebar_state='expanded')
# Application Header
st.title(" :bar_chart:  HDB Resale Price Evaluation")
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
    options=range(2000, datetime.now().year),
    index=(datetime.now().year - 2001)  # Default to current year
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
c1,c2,c3 = st.columns((3,4,4))
with c1:
    if 'prediction' in st.session_state:
        prediction = st.session_state['prediction']
        # Display the estimated valuation with increased font size
        st.markdown(f"""
            <h3 style="font-size: 20px; font-weight: bold;">Estimated valuation of home: ${round((prediction.item()), 2)}</h3>
        """, unsafe_allow_html=True)
        
        loan_tenure = st.slider("Select Loan Tenure (years):", min_value=1, max_value=35, value=20, step=1)
        grants = st.number_input("Enter CPF Housing Grant:", min_value=0, value=50000, step=1000)

        if st.button("Calculate Loan"):
            purchase_price = st.session_state['prediction']  # Example purchase price
            st.session_state['loan_details'] = calculate_hdb_loan_details(purchase_price, loan_tenure, grants)

with c2: 
    st.write("")
    
with c3:
    if 'loan_details' in st.session_state:
                # Assuming loan_details is available in the session state
        loan_details = st.session_state['loan_details']

        # Display loan details
        st.subheader("Loan Details")
        # Display loan details using the create_card function
        st.markdown(create_card("Purchase Price", f"${loan_details['purchase_price']}"), unsafe_allow_html=True)
        st.markdown(create_card("Loan Amount", f"${loan_details['loan_amount']}"), unsafe_allow_html=True)
        st.markdown(create_card("Tenure", f"{loan_details['tenure_years']} years"), unsafe_allow_html=True)
        st.markdown(create_card("Monthly Payment", f"${loan_details['monthly_payment']}"), unsafe_allow_html=True)
        st.markdown(create_card("Buyer's Stamp Duty (BSD)", f"${loan_details['bsd']}"), unsafe_allow_html=True)
