#pip show joblib
#pip show scikit-learn
#pip install joblib==1.0.0
#pip install scikit-learn==0.24

import streamlit as st
import pandas as pd
import joblib
import pickle
import os

# Load the trained model
model_path = "E:/Mandeep/360 DigiTMG/PROJECTS/TENDER PRICE OPTIMIZATION(1ST)/All Files/Model_Deployment/Deployment code(TPO)/DT_sklearn_model.pkl"

# Check if the model path exists
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}")
    st.stop()  # Stop the app if the model is not found

# Try loading with joblib, fallback to pickle if it fails
try:
    model = joblib.load(model_path)
    st.success("Model loaded successfully with joblib.")
except Exception as joblib_error:
    st.warning(f"Failed to load model with joblib: {joblib_error}")
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        st.success("Model loaded successfully with pickle.")
    except Exception as pickle_error:
        st.error(f"Failed to load model with pickle: {pickle_error}")
        st.stop()  # Stop the app if both loading methods fail

# Streamlit App
st.title("L1 Price Prediction")
st.write("Upload a CSV file to predict L1 Price.")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
    try:
        # Load the uploaded file
        input_data = pd.read_csv(uploaded_file)
        st.write("Preview of the uploaded data:")
        st.dataframe(input_data.head())

        # Check if the input data is empty or not structured as expected
        if input_data.empty:
            st.error("The uploaded CSV file is empty. Please upload a valid file.")
            st.stop()

        # Ensure the correct columns are present (you can modify this to match your model's expected features)
        expected_columns = ["feature1", "feature2", "feature3"]  # Modify with actual feature names
        if not all(col in input_data.columns for col in expected_columns):
            st.error(f"The uploaded CSV file is missing some expected columns. Expected columns: {expected_columns}")
            st.stop()

        # Predict L1 Price
        predictions = model.predict(input_data[expected_columns])  # Ensure to use only expected columns
        input_data["Predicted L1 Price"] = predictions

        # Display results
        st.write("Prediction Results:")
        st.dataframe(input_data)

        # Option to download results
        output_file = "predictions.csv"
        input_data.to_csv(output_file, index=False)
        with open(output_file, "rb") as f:
            st.download_button("Download Predictions", f, file_name=output_file)

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")



