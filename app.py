
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Mortality Risk Calculator for ICU Patients with Left Heart Failure",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load('SVC_best_model.pkl')
        scaler = joblib.load('selected_features_scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Model or scaler loading failed: {str(e)}")
        return None, None

st.title("Mortality Risk Calculator for ICU Patients with Left Heart Failure")
st.markdown("---")

model, scaler = load_model_and_scaler()
if model is None or scaler is None:
    st.stop()

features = ['PaO2', 'BUN', 'Age', 'Weight', 'LOS', 'PaCO2', 'RBC', 
            'APTT', 'SBP', 'Urine output (first 24h)', 'LVEF']

feature_ranges = {
    'PaO2': (0, 200, 'mmHg'),
    'BUN': (0, 200, 'mg/dL'),
    'Age': (18, 120, 'years'),
    'Weight': (0, 400, 'kg'),
    'LOS': (0, 100, 'days'),
    'PaCO2': (0, 200, 'mmHg'),
    'RBC': (0, 10, '×10¹²/L'),
    'APTT': (0, 200, 'seconds'),
    'SBP': (0, 300, 'mmHg'),
    'Urine output (first 24h)': (0, 10000, 'mL'),
    'LVEF': (0, 100, '%')
}

col1, col2 = st.columns([2, 1])

with col1:
    st.header("Patient Information Input")
    
    with st.form("patient_form"):
        input_col1, input_col2 = st.columns(2)
        
        input_values = {}
        
        with input_col1:
            for i, feature in enumerate(features[:6]):
                min_val, max_val, unit = feature_ranges[feature]
                input_values[feature] = st.number_input(
                    f"{feature} ({unit})",
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float((min_val + max_val) / 2),
                    step=0.1
                )
        
        with input_col2:
            for i, feature in enumerate(features[6:]):
                min_val, max_val, unit = feature_ranges[feature]
                input_values[feature] = st.number_input(
                    f"{feature} ({unit})",
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float((min_val + max_val) / 2),
                    step=0.1
                )
        
        submitted = st.form_submit_button("Predict")
        
        if submitted:
            input_data = [input_values[feature] for feature in features]
            
            input_df = pd.DataFrame([input_data], columns=features)
            
            try:
                # 对输入数据进行标准化
                input_scaled = scaler.transform(input_df)
                
                probability = model.predict_proba(input_scaled)[0][1] * 100
                st.success("Prediction completed!")
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

with col2:
    st.header("Prediction Results")
    
    if submitted:
        st.metric(
            label="In-hospital Mortality Probability",
            value=f"{probability:.1f}%"
        )
        
        st.info("The mortality rate for ICU patients with left heart failure in the MIMIC-IV database used for model development was 5.5%, for reference by clinical decision-makers.")
    else:
        st.info("Please fill in the patient information and click 'Predict' to see results.")

st.markdown("---")
st.header("About This Calculator")
st.markdown("""
This Mortality Risk Calculator for ICU Patients with Left Heart Failure uses a machine learning model trained on clinical data to assess mortality risk based on 11 key patient parameters.

**Important Note**: All input features are automatically standardized using the same scaler that was used during model training to ensure prediction accuracy.

Parameters Used:
- PaO2: Partial pressure of arterial oxygen
- BUN: Blood urea nitrogen
- Age: Patient age
- Weight: Patient weight
- LOS: Length of stay in ICU
- PaCO2: Partial pressure of arterial carbon dioxide
- RBC: Red blood cell count
- APTT: Activated partial thromboplastin time
- SBP: Systolic blood pressure
- Urine output (first 24h): Urine output in the first 24 hours
- LVEF: Left ventricular ejection fraction
""")

with st.sidebar:
    st.header("Mortality Risk Calculator")
    st.markdown("""
    This application predicts mortality risk for ICU patients with left heart failure based on clinical parameters.
    
    **Data Processing**:
    - All input features are automatically standardized
    - Uses the same preprocessing as model training
    
    Instructions:
    1. Fill in all patient parameters
    2. Click the 'Predict' button
    3. View the prediction results
    
    Note: Ensure all values are within the specified ranges.
    """)
    
    if st.button("Reset Form"):
        st.rerun()
