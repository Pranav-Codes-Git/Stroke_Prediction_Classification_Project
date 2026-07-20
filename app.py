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

    # If your model was trained on SCALED features, you must load and use
    # the same scaler here, or predictions will be unreliable/stuck.
    scaler = None
    try:
        with open("scaler.pkl", "rb") as s_file:
            scaler = pickle.load(s_file)
    except FileNotFoundError:
        pass

    return model, scaler

model, scaler = load_assets()

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
gender_enc = 1 if gender == "Male" else 0
hypertension_enc = 1 if hypertension == "Yes" else 0
heart_disease_enc = 1 if heart_disease == "Yes" else 0
ever_married_enc = 1 if ever_married == "Yes" else 0
Residence_type_enc = 1 if Residence_type == "Urban" else 0

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
    "gender": [gender_enc],
    "age": [age],
    "hypertension": [hypertension_enc],
    "heart_disease": [heart_disease_enc],
    "ever_married": [ever_married_enc],
    "Residence_type": [Residence_type_enc],
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
expected_cols = None
missing_from_input = []
dropped_from_input = []
if hasattr(model, "feature_names_in_"):
    expected_cols = list(model.feature_names_in_)
    built_cols = list(input_data.columns)
    missing_from_input = [c for c in expected_cols if c not in built_cols]
    dropped_from_input = [c for c in built_cols if c not in expected_cols]
    input_data = input_data.reindex(columns=expected_cols, fill_value=0)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict Stroke"):

    model_input = input_data
    if scaler is not None:
        model_input = pd.DataFrame(
            scaler.transform(input_data),
            columns=input_data.columns
        )

    # Confirm which column index actually corresponds to the "stroke" class.
    # For a binary classifier this is usually 1, but we check model.classes_
    # to be safe instead of assuming.
    classes = list(model.classes_) if hasattr(model, "classes_") else [0, 1]
    stroke_class = 1
    stroke_index = classes.index(stroke_class) if stroke_class in classes else 1

    probability = model.predict_proba(model_input)[0][stroke_index]
    prediction = 1 if probability >= 0.5 else 0

    st.subheader("Prediction")

    if prediction == 1:
        st.error("⚠️ High Risk of Stroke")
    else:
        st.success("✅ Low Risk of Stroke")

    # Debugging helper
    with st.expander("🔍 Debug Input Data"):
        st.write("Model classes:", classes)
        st.write("Scaler applied:", scaler is not None)
        st.write("Model's expected feature names:", expected_cols)
        if missing_from_input:
            st.warning(
                f"These columns the model expects were NOT built by the app, "
                f"and got silently filled with 0: {missing_from_input}"
            )
        if dropped_from_input:
            st.warning(
                f"These columns the app built were NOT expected by the model, "
                f"and got silently dropped: {dropped_from_input}"
            )
        st.dataframe(model_input)

st.markdown("---")
st.caption("Stroke Prediction using Logistic Regression")
st.markdown("---")
st.caption("Stroke Prediction using Logistic Regression")
