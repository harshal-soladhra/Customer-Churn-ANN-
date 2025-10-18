import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import streamlit as st

# Load model
model = tf.keras.models.load_model("model.h5")

# Load encoder and scaler
with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("sclare.pkl", "rb") as file:
    scaler = pickle.load(file)

with open("onehot_encoder_geo.pkl", "rb") as file:
    onehot_encoder_geo = pickle.load(file)

# Streamlit app
st.title("Customer Churn Prediction")

# Input fields for all features
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 18, 92, 40)
tenure = st.slider("Tenure (Years with Bank)", 0, 10, 3)
balance = st.number_input("Account Balance", min_value=0.0, value=60000.0, step=1000.0)
num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.selectbox("Has Credit Card?", [1, 0])
is_active_member = st.selectbox("Is Active Member?", [1, 0])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0, step=1000.0)

# Create a dictionary for the input (now includes Geography)
input_data = {
    "CreditScore": [credit_score],
    "Geography": [geography],
    "Gender": [label_encoder_gender.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]
}

# Convert to DataFrame
input_df = pd.DataFrame(input_data)

# One-hot encode Geography
geo_encoded = onehot_encoder_geo.transform(input_df[["Geography"]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(["Geography"]))

# Merge encoded columns with the rest of the features
input_df = pd.concat([input_df.drop("Geography", axis=1), geo_encoded_df], axis=1)

# Scale the input data
input_scaled = scaler.transform(input_df)

# Predict churn probability
prediction = model.predict(input_scaled)
prediction_proba = prediction[0][0]

# Show result in Streamlit
if prediction_proba > 0.5:
    st.error(f"🚨 The customer is likely to churn with a probability of **{prediction_proba:.2f}**")
else:
    st.success(f"✅ The customer is unlikely to churn with a probability of **{prediction_proba:.2f}**")
