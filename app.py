import streamlit as st
import joblib
import numpy as np

# Load the model and scaler
model = joblib.load('fraud_detection_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit app
st.title("Fraud Detection in Financial Transactions")

# Input fields for features
st.header("Enter Transaction Details")
input_data = []
for i in range(1, 29):
    input_data.append(st.number_input(f"V{i}", value=0.0))

amount = st.number_input("Amount", value=0.0)

# Preprocess the input data
input_data = np.array(input_data).reshape(1, -1)
amount_scaled = scaler.transform([[amount]])

# Combine features
features = np.hstack((input_data, amount_scaled))

# Make prediction
if st.button("Predict"):
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.error("Fraudulent Transaction Detected!")
    else:
        st.success("Legitimate Transaction.")