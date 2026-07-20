import streamlit as st
import pandas as pd
import pickle

# ----------------------------
# Load Model & Scaler
# ----------------------------
@st.cache_resource
def load_assets():
    with open("stroke_prediction_model.pkl", "rb") as file:
        model = pickle.load(file)
    
    # Optional: Load your saved scaler if trained separately
    # with open("scaler.pkl", "rb") as s_file:
    #     scaler = pickle.load(s_file)
    # return model, scaler
    
    return model

model = load_assets()

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(page_title="Stroke Prediction", page_icon="🩺")
st.title("🧠 Stroke Prediction App")
st.write("Enter patient details below to predict stroke risk.")

# ----------------------------
# User Inputs
# ----------------------------
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 1, 100, 30)

hypertension = st.selectbox("Hypertension", ["No", "Yes"])
heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
ever_married = st.selectbox("Ever Married", ["No", "Yes"])

work_type = st.selectbox(
    "Work Type",
    ["Private", "Self-employed", "Govt_job", "children", "Never_worked"]
)

Residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])

avg_glucose_level = st.number_input(
    "Average Glucose Level", min_value=50.0, max_value=300.0, value=100.0
)

bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0)

smoking_status = st.selectbox(
    "Smoking Status",
    ["never smoked", "formerly smoked", "smokes", "Unknown"]
)

# ----------------------------
# Encoding
# ----------------------------
gender = 1 if gender == "Male" else 0
hypertension = 1 if hypertension == "Yes" else 0
heart_disease = 1 if heart_disease == "Yes" else 0
ever_married = 1 if ever_married == "Yes" else 0
Residence_type = 1 if Residence_type == "Urban" else 0

# Work Type One-Hot Encoding
work_type_Never_worked = 1 if work_type == "Never_worked" else 0
work_type_Private = 1 if work_type == "Private" else 0
work_type_Self_employed = 1 if work_type == "Self-employed" else 0
work_type_children = 1 if work_type == "children" else 0

# Smoking Status One-Hot Encoding
smoking_status_formerly_smoked = 1 if smoking_status == "formerly smoked" else 0
smoking_status_never_smoked = 1 if smoking_status == "never smoked" else 0
smoking_status_smokes = 1 if smoking_status == "smokes" else 0

# ----------------------------
# Create Input DataFrame
# ----------------------------
input_data = pd.DataFrame({
    "gender": [gender],
    "age": [age],
    "hypertension": [hypertension],
    "heart_disease": [heart_disease],
    "ever_married": [ever_married],
    "Residence_type": [Residence_type],
    "avg_glucose_level": [avg_glucose_level],
    "bmi": [bmi],
    "work_type_Never_worked": [work_type_Never_worked],
    "work_type_Private": [work_type_Private],
    "work_type_Self-employed": [work_type_Self_employed],
    "work_type_children": [work_type_children],
    "smoking_status_formerly smoked": [smoking_status_formerly_smoked],
    "smoking_status_never smoked": [smoking_status_never_smoked],
    "smoking_status_smokes": [smoking_status_smokes]
})

# Ensure columns match model expectation if feature_names_in_ exists
if hasattr(model, "feature_names_in_"):
    input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict Stroke"):
    # Apply scaler here if using a standalone scaler object:
    # scaled_input = scaler.transform(input_data)
    # probability = model.predict_proba(scaled_input)[0][1]
    
    probability = model.predict_proba(input_data)[0][1]
    
    # Threshold check (0.5 or custom threshold)
    prediction = 1 if probability >= 0.5 else 0

    st.subheader("Prediction")

    if prediction == 1:
        st.error("⚠️ High Risk of Stroke")
    else:
        st.success("✅ Low Risk of Stroke")
    # Debugging helper
    with st.expander("🔍 Debug Input Data"):
        st.dataframe(input_data)
st.markdown("---")
st.caption("Stroke Prediction using Logistic Regression")
