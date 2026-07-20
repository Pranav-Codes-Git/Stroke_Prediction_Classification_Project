import streamlit as st
import pandas as pd
import pickle

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------
st.set_page_config(
    page_title="Stroke Risk Prediction Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------
# LOAD MODEL
# -----------------------------------------------------
@st.cache_resource
def load_assets():
    with open("stroke_prediction_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_assets()

# -----------------------------------------------------
# CUSTOM CSS
# -----------------------------------------------------
st.markdown("""
<style>

.main{
    background-color:#f5f7fa;
}

h1{
    color:#1565C0;
    text-align:center;
}

h3{
    color:#0D47A1;
}

div.stButton > button:first-child{
    background:#1565C0;
    color:white;
    height:55px;
    width:100%;
    font-size:18px;
    border-radius:12px;
    border:none;
}

div.stButton > button:first-child:hover{
    background:#0D47A1;
    color:white;
}

[data-testid="stMetric"]{
    background:white;
    padding:15px;
    border-radius:12px;
    box-shadow:0px 2px 8px rgba(0,0,0,0.15);
}

div[data-testid="stVerticalBlock"]{
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# SIDEBAR
# -----------------------------------------------------
with st.sidebar:

    st.title("🩺 Stroke Prediction")

    st.markdown("---")

    st.info("""
This dashboard predicts the **risk of stroke**
using a trained **Logistic Regression** model.

Fill in the patient's information and click
**Predict Stroke Risk**.
""")

    st.markdown("### 📊 Model")

    st.success("Logistic Regression")

    st.markdown("---")

    st.markdown("### 👨‍⚕️ Tips")

    st.write("• Maintain healthy BMI")
    st.write("• Exercise regularly")
    st.write("• Avoid smoking")
    st.write("• Monitor blood pressure")
    st.write("• Eat a balanced diet")

# -----------------------------------------------------
# HEADER
# -----------------------------------------------------
st.markdown("""
# 🧠 Stroke Risk Prediction Dashboard

### Predict the likelihood of stroke using Machine Learning

---
""")

# -----------------------------------------------------
# INPUT FORM
# -----------------------------------------------------
left, right = st.columns(2)

with left:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    age = st.slider(
        "Age",
        1,
        100,
        30
    )

    hypertension = st.selectbox(
        "Hypertension",
        ["No", "Yes"]
    )

    heart_disease = st.selectbox(
        "Heart Disease",
        ["No", "Yes"]
    )

    ever_married = st.selectbox(
        "Ever Married",
        ["No", "Yes"]
    )

with right:

    work_type = st.selectbox(
        "Work Type",
        [
            "Private",
            "Self-employed",
            "Govt_job",
            "children",
            "Never_worked"
        ]
    )

    Residence_type = st.selectbox(
        "Residence Type",
        ["Urban", "Rural"]
    )

    avg_glucose_level = st.number_input(
        "Average Glucose Level",
        min_value=50.0,
        max_value=300.0,
        value=100.0
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=70.0,
        value=25.0
    )

    smoking_status = st.selectbox(
        "Smoking Status",
        [
            "never smoked",
            "formerly smoked",
            "smokes",
            "Unknown"
        ]
    )

st.markdown("---")

# -----------------------------------------------------
# ENCODING
# -----------------------------------------------------
gender = 1 if gender == "Male" else 0

hypertension = 1 if hypertension == "Yes" else 0

heart_disease = 1 if heart_disease == "Yes" else 0

ever_married = 1 if ever_married == "Yes" else 0

Residence_type = 1 if Residence_type == "Urban" else 0

# Work Type Encoding
work_type_Never_worked = 1 if work_type == "Never_worked" else 0
work_type_Private = 1 if work_type == "Private" else 0
work_type_Self_employed = 1 if work_type == "Self-employed" else 0
work_type_children = 1 if work_type == "children" else 0

# Smoking Status Encoding
smoking_status_formerly_smoked = 1 if smoking_status == "formerly smoked" else 0
smoking_status_never_smoked = 1 if smoking_status == "never smoked" else 0
smoking_status_smokes = 1 if smoking_status == "smokes" else 0

# -----------------------------------------------------
# CREATE INPUT DATAFRAME
# -----------------------------------------------------
input_data = pd.DataFrame({

    "gender":[gender],
    "age":[age],
    "hypertension":[hypertension],
    "heart_disease":[heart_disease],
    "ever_married":[ever_married],
    "Residence_type":[Residence_type],
    "avg_glucose_level":[avg_glucose_level],
    "bmi":[bmi],
    "work_type_Never_worked":[work_type_Never_worked],
    "work_type_Private":[work_type_Private],
    "work_type_Self-employed":[work_type_Self_employed],
    "work_type_children":[work_type_children],
    "smoking_status_formerly smoked":[smoking_status_formerly_smoked],
    "smoking_status_never smoked":[smoking_status_never_smoked],
    "smoking_status_smokes":[smoking_status_smokes]

})

if hasattr(model, "feature_names_in_"):
    input_data = input_data.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )
st.markdown("---")
st.caption("Stroke Prediction using Logistic Regression")
# -----------------------------------------------------
# PATIENT SUMMARY
# -----------------------------------------------------
st.subheader("📋 Patient Summary")

summary_col1, summary_col2, summary_col3 = st.columns(3)

with summary_col1:
    st.metric("Age", age)
    st.metric("BMI", bmi)

with summary_col2:
    st.metric("Glucose", avg_glucose_level)
    st.metric("Gender", "Male" if gender else "Female")

with summary_col3:
    st.metric("Hypertension", "Yes" if hypertension else "No")
    st.metric("Heart Disease", "Yes" if heart_disease else "No")

st.markdown("---")

# -----------------------------------------------------
# PREDICTION
# -----------------------------------------------------
if st.button("🩺 Predict Stroke Risk"):

    probability = model.predict_proba(input_data)[0][1]

    # Change threshold if desired
    prediction = probability >= 0.5

    st.subheader("Prediction Result")
    # Progress Bar
    st.progress(float(probability))

    st.write("")

    # Risk Level
    if probability < 0.30:

        st.success("🟢 **LOW RISK OF STROKE**")

    elif probability < 0.60:

        st.warning("🟡 **MODERATE RISK OF STROKE**")

    else:

        st.error("🔴 **HIGH RISK OF STROKE**")

    st.markdown("---")

    # -------------------------------------------------
    # Recommendations
    # -------------------------------------------------

    st.subheader("💡 Health Recommendations")

    if probability < 0.30:

        st.success("""
✅ Continue maintaining a healthy lifestyle.

• Exercise regularly

• Maintain a healthy weight

• Eat fruits and vegetables

• Drink enough water

• Get annual health check-ups
""")

    elif probability < 0.60:

        st.warning("""
⚠ Moderate risk detected.

• Monitor blood pressure

• Reduce sugar intake

• Exercise 30 minutes daily

• Improve sleep quality

• Schedule regular doctor visits
""")

    else:

        st.error("""
🚨 High stroke risk detected.

• Consult a doctor immediately

• Monitor blood pressure frequently

• Control blood sugar

• Avoid smoking and alcohol

• Follow prescribed medication

• Maintain a healthy BMI

• Exercise only after medical advice
""")

    # -------------------------------------------------
    # Patient Input Table
    # -------------------------------------------------

    with st.expander("📄 View Encoded Input Data"):

        st.dataframe(input_data)

st.markdown("---")

# -----------------------------------------------------
# FOOTER
# -----------------------------------------------------

st.markdown(
"""
<div style="text-align:center; color:gray;">

### 🧠 Stroke Prediction Dashboard

Developed using **Streamlit**, **Python**, **Pandas**, and **Scikit-Learn**

Machine Learning Model: **Logistic Regression**

<i>This prediction is intended for educational purposes only and should not replace professional medical advice.</i>

</div>
""",
unsafe_allow_html=True
)
