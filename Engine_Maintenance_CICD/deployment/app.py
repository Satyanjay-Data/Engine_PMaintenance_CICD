import streamlit as st
import pandas as pd
import joblib
import os
from huggingface_hub import hf_hub_download

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    try:
        # Download model from Hugging Face
        model_path = hf_hub_download(
            repo_id="Satyanjay/engine-condition-monitoring-model",
            filename="best_model.joblib"
        )
    except:
        # fallback if running locally
        model_path = "best_model.joblib"

    model = joblib.load(model_path)
    return model


model = load_model()

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="Engine Condition Monitoring", layout="centered")

st.title("🚗 Engine Condition Monitoring System")
st.write("Enter engine parameters below to predict condition.")

# Input fields
engine_rpm = st.number_input("Engine RPM", min_value=0.0)
lub_oil_pressure = st.number_input("Lub Oil Pressure", min_value=0.0)
fuel_pressure = st.number_input("Fuel Pressure", min_value=0.0)
coolant_pressure = st.number_input("Coolant Pressure", min_value=0.0)
lub_oil_temp = st.number_input("Lub Oil Temperature", min_value=0.0)
coolant_temp = st.number_input("Coolant Temperature", min_value=0.0)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict"):

    input_data = pd.DataFrame([{
        'Engine rpm': engine_rpm,
        'Lub oil pressure': lub_oil_pressure,
        'Fuel pressure': fuel_pressure,
        'Coolant pressure': coolant_pressure,
        'Lub oil temp': lub_oil_temp,
        'Coolant temp': coolant_temp
    }])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ Engine Condition: FAULT DETECTED")
    else:
        st.success("✅ Engine Condition: NORMAL")
