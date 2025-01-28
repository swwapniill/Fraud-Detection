import streamlit as st
import joblib
import numpy as np

# Load the model and scaler
model = joblib.load('fraud_detection_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit app
st.title("Fraud Detection in Financial Transactions")
st.write("""
This app predicts whether a financial transaction is fraudulent or legitimate based on transaction details.
""")

# Input fields for features
st.header("Enter Transaction Details")

# Real-world inputs
amount = st.number_input("Transaction Amount", value=0.0, min_value=0.0, step=0.01)
time = st.number_input("Time of Transaction (in seconds since the first transaction)", value=0, min_value=0)

# Anonymized features (V1-V28)
st.subheader("Anonymized Features (V1-V28)")
st.write("""
These features are anonymized for privacy reasons. Enter values between -10 and 10.
""")

input_data = []
for i in range(1, 29):
    input_data.append(st.number_input(f"V{i}", value=0.0, min_value=-10.0, max_value=10.0))

# Preprocess the input data
input_data = np.array(input_data).reshape(1, -1)
amount_scaled = scaler.transform([[amount]])

# Combine features
features = np.hstack((input_data, amount_scaled))

# Make prediction
if st.button("Predict"):
    prediction = model.predict(features)
    prediction_proba = model.predict_proba(features)

    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction.")

    st.write(f"Probability of Fraud: {prediction_proba[0][1]:.2%}")
    st.write(f"Probability of Legitimacy: {prediction_proba[0][0]:.2%}")

# Add instructions
st.sidebar.header("Instructions")
st.sidebar.write("""
1. Enter the **Transaction Amount** and **Time**.
2. For the anonymized features (V1-V28), enter values between -10 and 10.
3. Click **Predict** to see the result.
""")
