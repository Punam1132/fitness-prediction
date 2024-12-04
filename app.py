import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the pre-trained model
model = joblib.load("xgb_model.pkl")

# Title of the app
st.title("Fitness Activity Prediction App")
st.write("This app predicts the outcome based on fitness tracker data.")

# User input form
st.header("Input Features")

# Example input features based on fitness tracker data
Accelerometer_x = st.number_input("Accelerometer X", value=0.0, step=0.1)
Accelerometer_y = st.number_input("Accelerometer Y", value=0.0, step=0.1)
Accelerometer_z = st.number_input("Accelerometer Z", value=0.0, step=0.1)
Gyroscope_x = st.number_input("Gyroscope X", value=0.0, step=0.1)
Gyroscope_y = st.number_input("Gyroscope Y", value=0.0, step=0.1)
Gyroscope_z = st.number_input("Gyroscope Z", value=0.0, step=0.1)

# Create a DataFrame for the input data
input_data = pd.DataFrame({
    "Accelerometer_x": [Accelerometer_x],
    "Accelerometer_y": [Accelerometer_y],
    "Accelerometer_z": [Accelerometer_z],
    "Gyroscope_x": [Gyroscope_x],
    "Gyroscope_y": [Gyroscope_y],
    "Gyroscope_z": [Gyroscope_z],
})

st.write("## Input Data")
st.dataframe(input_data)

# Prediction button
if st.button("Predict"):
    # Make predictions
    prediction = model.predict(input_data)
    st.success(f"The predicted value is: {prediction[0]:.2f}")
