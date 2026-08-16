import streamlit as st
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..','src'))
from predict import predict_heart_disease

st.title("Heart Disease Risk Prediction")

st.header("Patient Information")

age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", options=["M", "F"])
chest_pain_type = st.selectbox("Chest Pain Type", options=["ATA", "NAP", "ASY", "TA"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=250, value=120)
cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=0, max_value=700, value=200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1])
resting_ecg = st.selectbox("Resting ECG", options=["Normal", "ST", "LVH"])
max_hr = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
exercise_angina = st.selectbox("Exercise-Induced Angina", options=["Y", "N"])
oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=-3.0, max_value=7.0, value=0.0, step=0.1)
st_slope = st.selectbox("ST Slope", options=["Up", "Flat", "Down"])

if st.button("Predict"):
    patient_data = {'Age': age,
                    'Sex': sex, 
                    'ChestPainType': chest_pain_type,
                    'RestingBP': resting_bp,
                    'Cholesterol' : cholesterol,
                    'FastingBS': fasting_bs,
                    'RestingECG': resting_ecg,
                    'MaxHR': max_hr,
                    'ExerciseAngina': exercise_angina,
                    'Oldpeak' : oldpeak,
                    'ST_Slope': st_slope,
    }
    result = predict_heart_disease(patient_data)

# Threshold (0.35) selected during evaluation to prioritize recall
    # over precision, given the asymmetric cost of false negatives
    if result['prediction'] == 1:
        st.error(f"High Risk of Heart Disease - Probability: {result['probability']:.2f}")
    else:
        st.success(f"Low Risk of Heart Disease - Probability: {result['probability']:.2f}")


    st.caption("Note: This prediction is based on a machine learning model and is for "
        "educational purposes only, and should not replace professional medical "
        "advice. Please consult a healthcare provider for a comprehensive evaluation."
    )