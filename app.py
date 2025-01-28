import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Set page title and icon
st.set_page_config(page_title="Fraud Detection App", page_icon="🕵️")

# Add a title and description
st.title("Fraud Detection in Financial Transactions")
st.markdown("""
    This app detects fraudulent transactions using a machine learning model. 
    Upload your dataset or use the sample dataset provided to see how it works!
""")

# Sidebar for user inputs
st.sidebar.header("User Input")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
use_sample_data = st.sidebar.checkbox("Use sample dataset", value=True)

# Load data
@st.cache_data
def load_data():
    # Sample dataset (replace with your own dataset)
    data = pd.read_csv('creditcard.csv')  # Ensure you have a sample dataset
    return data

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
elif use_sample_data:
    data = load_data()
else:
    st.warning("Please upload a file or use the sample dataset.")
    st.stop()

# Display dataset
st.subheader("Dataset Preview")
st.write(data.head())

# Data preprocessing
st.subheader("Data Preprocessing")
st.write("Handling missing values and feature engineering...")

# Example: Drop missing values and create a binary target column
data = data.dropna()
data['is_fraud'] = data['is_fraud'].astype(int)

# Show class distribution
st.write("Class Distribution (Fraud vs Non-Fraud):")
fraud_counts = data['is_fraud'].value_counts()
st.bar_chart(fraud_counts)

# Feature selection
features = data.drop(columns=['is_fraud'])
target = data['is_fraud']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Model training
st.subheader("Model Training")
st.write("Training a Random Forest Classifier...")

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Model evaluation
st.subheader("Model Evaluation")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Confusion Matrix
st.write("Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)

# Classification Report
st.write("Classification Report:")
st.text(classification_report(y_test, y_pred))

# ROC Curve
st.write("ROC Curve:")
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
ax.plot([0, 1], [0, 1], linestyle='--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend(loc='lower right')
st.pyplot(fig)

# Feature Importance
st.subheader("Feature Importance")
importances = model.feature_importances_
feature_names = features.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
st.bar_chart(importance_df.set_index('Feature'))

# Prediction Interface
st.subheader("Make Predictions")
st.write("Enter transaction details to predict if it's fraudulent:")

# Example input fields (customize based on your dataset)
transaction_amount = st.number_input("Transaction Amount", min_value=0.0)
transaction_time = st.number_input("Transaction Time (in hours)", min_value=0.0)
user_balance = st.number_input("User Balance", min_value=0.0)

if st.button("Predict"):
    input_data = np.array([[transaction_amount, transaction_time, user_balance]])
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.error("This transaction is predicted to be **fraudulent**.")
    else:
        st.success("This transaction is predicted to be **legitimate**.")

    st.write(f"Prediction Probability: {prediction_proba[0][1]:.2f}")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")
