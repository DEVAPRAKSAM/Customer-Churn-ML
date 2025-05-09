import streamlit as st
import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# Load model
model = joblib.load("xgboost_churn_model.pkl")

# Title
st.title("🔍 Customer Churn Prediction App")

# Input form
st.header("Enter Customer Details:")

gender = st.selectbox("Gender", ["Female", "Male"])
senior = st.selectbox("Senior Citizen", ["No", "Yes"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.slider("Monthly Charges", 0, 150, 50)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

# Preprocessing
gender = 1 if gender == "Male" else 0
senior = 1 if senior == "Yes" else 0
contract_one_year = 1 if contract == "One year" else 0
contract_two_year = 1 if contract == "Two year" else 0

# Make feature array
features = np.array([[tenure, monthly_charges, senior, gender, contract_one_year, contract_two_year]])

# Predict
if st.button("Predict Churn"):
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.error("⚠️ This customer is likely to CHURN.")
    else:
        st.success("✅ This customer is likely to STAY.")

# 1. Dataset load pannunga (replace with your file name)
df = pd.read_csv("customer_churn.csv")

# 2. Split features & target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train model
model = XGBClassifier()
model.fit(X_train, y_train)

# 5. Save as .pkl
joblib.dump(model, "xgboost_churn_model.pkl")

print("✅ Model trained and saved successfully!")