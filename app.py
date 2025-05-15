import streamlit as st
import numpy as np
import joblib

# Load the model
model = joblib.load("xgboost_churn_model.pkl")

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("📉 Customer Churn Prediction App")
st.markdown("Enter the customer details below to predict whether they are likely to churn.")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Is the customer a senior citizen?", ["Yes", "No"])
partner = st.selectbox("Has a partner?", ["Yes", "No"])
dependents = st.selectbox("Has dependents?", ["Yes", "No"])
tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)
monthly_charges = st.slider("Monthly Charges", min_value=0.0, max_value=150.0, value=50.0)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No"])
tech_support = st.selectbox("Tech Support", ["Yes", "No"])
paperless_billing = st.selectbox("Uses Paperless Billing?", ["Yes", "No"])
payment_method = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])

# Feature engineering
gender_male = 1 if gender == "Male" else 0
senior_citizen = 1 if senior_citizen == "Yes" else 0
partner = 1 if partner == "Yes" else 0
dependents = 1 if dependents == "Yes" else 0
paperless_billing = 1 if paperless_billing == "Yes" else 0

# One-hot for contract
contract_one_year = 1 if contract == "One year" else 0
contract_two_year = 1 if contract == "Two year" else 0

# One-hot for internet service
internet_fiber = 1 if internet_service == "Fiber optic" else 0
internet_no = 1 if internet_service == "No" else 0

# Online security and tech support
online_security = 1 if online_security == "Yes" else 0
tech_support = 1 if tech_support == "Yes" else 0

# One-hot for payment method
payment_bank = 1 if payment_method == "Bank transfer (automatic)" else 0
payment_credit = 1 if payment_method == "Credit card (automatic)" else 0
payment_mailed = 1 if payment_method == "Mailed check" else 0

# Prepare input
features = np.array([[tenure, monthly_charges, senior_citizen, gender_male, partner,
                      dependents, paperless_billing, contract_one_year, contract_two_year,
                      internet_fiber, internet_no, online_security, tech_support,
                      payment_bank, payment_credit, payment_mailed]])

# Predict button
if st.button("🔍 Predict"):
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.error("⚠️ This customer is likely to CHURN.")
    else:
        st.success("✅ This customer is likely to STAY.")
