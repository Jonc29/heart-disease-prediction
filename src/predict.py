import joblib
import pandas as pd

# Load trained artifacts once at module import time
model = joblib.load('models/logistic_model.pkl')
scaler = joblib.load('models/scaler.pkl')
model_columns = joblib.load('models/model_columns.pkl')

DECISION_THRESHOLD = 0.35
NUMERIC_COLS = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

def preprocess_patient(patient_dict):
    """Convert raw patient input into a model-ready feature vector."""
    
    df = pd.DataFrame([patient_dict])
    
    # Label encoding for binary categorical fields
    df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})
    df['ExerciseAngina'] = df['ExerciseAngina'].map({'Y': 1, 'N': 0})
    
    # One-hot encode multi-category fields
    df = pd.get_dummies(df, columns=['ChestPainType', 'RestingECG', 'ST_Slope'])
    
    # Align columns to match training feature set exactly
    df = df.reindex(columns=model_columns, fill_value=0)
    
    # Scale numeric columns using the training-fitted scaler
    df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])
    
    return df


def predict_heart_disease(patient_dict):
    """Predict heart disease risk for a single patient using the tuned decision threshold."""
    
    X_processed = preprocess_patient(patient_dict)
    probability = model.predict_proba(X_processed)[:, 1][0]
    prediction = int(probability >= DECISION_THRESHOLD)
    
    return {
        'prediction': prediction,
        'probability': round(float(probability), 4)
    }


if __name__ == "__main__":
    high_risk_patient = {
        'Age': 58, 'Sex': 'M', 'ChestPainType': 'ASY', 'RestingBP': 145,
        'Cholesterol': 233, 'FastingBS': 1, 'RestingECG': 'Normal',
        'MaxHR': 110, 'ExerciseAngina': 'Y', 'Oldpeak': 2.3, 'ST_Slope': 'Flat'
    }

    low_risk_patient = {
        'Age': 35, 'Sex': 'F', 'ChestPainType': 'ATA', 'RestingBP': 115,
        'Cholesterol': 190, 'FastingBS': 0, 'RestingECG': 'Normal',
        'MaxHR': 175, 'ExerciseAngina': 'N', 'Oldpeak': 0.0, 'ST_Slope': 'Up'
    }

    print("High-risk case:", predict_heart_disease(high_risk_patient))
    print("Low-risk case:", predict_heart_disease(low_risk_patient))