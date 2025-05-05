import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/Random_Forest.pkl")

# Title
st.title("Breast Cancer Prediction App")

st.write("Please input the following tumor measurements:")

# Input form
radius_mean = st.number_input("Radius Mean",  format="%.2f")
concavity_mean = st.number_input("Concavity Mean",  format="%.2f")
smoothness_mean = st.number_input("Smoothness Mean",  format="%.4f")
texture_mean = st.number_input("Texture Mean", format="%.2f")

if st.button("Predict"):
    # Create DataFrame from inputs
    input_data = pd.DataFrame([{
        "radius_mean": radius_mean,
        "concavity_mean": concavity_mean,
        "smoothness_mean": smoothness_mean,
        "texture_mean": texture_mean
    }])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Show result
    if prediction == 1:
        st.success("Prediction: Malignant")
    else:
        st.info("Prediction: Benign")
