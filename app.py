import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("xgboost_churn_model.pkl")

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("📉 Customer Churn Prediction App")
st.markdown("Fill the details below to check if a customer is likely to churn.")

# User Inputs
gender = st.selectbox("Gender", ["Female", "Male"])
senior_citizen = st.selectbox("Senior Citizen?", ["No", "Yes"])
partner = st.selectbox("Has Partner?", ["No", "Yes"])
dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
tenure = st.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.slider("Monthly Charges", 0.0, 150.0, 70.0)
paperless_billing = st.selectbox("Uses Paperless Billing?", ["No", "Yes"])
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["No", "Yes"])
tech_support = st.selectbox("Tech Support", ["No", "Yes"])
payment_method = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])

# Feature Engineering (One-hot encode like in training)
features = {
    'tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
    'gender_Male': 1 if gender == "Male" else 0,
    'Partner_Yes': 1 if partner == "Yes" else 0,
    'Dependents_Yes': 1 if dependents == "Yes" else 0,
    'PaperlessBilling_Yes': 1 if paperless_billing == "Yes" else 0,
    'Contract_One year': 1 if contract == "One year" else 0,
    'Contract_Two year': 1 if contract == "Two year" else 0,
    'InternetService_Fiber optic': 1 if internet_service == "Fiber optic" else 0,
    'InternetService_No': 1 if internet_service == "No" else 0,
    'OnlineSecurity_Yes': 1 if online_security == "Yes" else 0,
    'TechSupport_Yes': 1 if tech_support == "Yes" else 0,
    'PaymentMethod_Bank transfer (automatic)': 1 if payment_method == "Bank transfer (automatic)" else 0,
    'PaymentMethod_Credit card (automatic)': 1 if payment_method == "Credit card (automatic)" else 0,
    'PaymentMethod_Mailed check': 1 if payment_method == "Mailed check" else 0,
}

# Create a DataFrame with the correct column structure
input_df = pd.DataFrame([features])

# Prediction
if st.button("🔍 Predict"):
    prediction = model.predict(input_df)
    if prediction[0] == 1:
        st.error("⚠️ This customer is likely to CHURN.")
    else:
        st.success("✅ This customer is likely to STAY.")
