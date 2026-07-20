# 🧠 Stroke Prediction Using Machine Learning

A Machine Learning project that predicts the likelihood of a person having a stroke based on demographic, lifestyle, and medical attributes. This project includes data preprocessing, exploratory data analysis (EDA), feature engineering, model training, evaluation, and a Streamlit web application for real-time predictions.

---

## 📌 Project Overview

Stroke is one of the leading causes of death and long-term disability worldwide. Early prediction can help healthcare professionals identify high-risk patients and take preventive measures.

This project develops a predictive machine learning model using patient health records to estimate stroke risk.

---

## 🎯 Business Objective

The objective of this project is to build a machine learning model capable of predicting whether a patient is at risk of stroke.

Healthcare organizations can use this model to:

- Identify high-risk patients early
- Support preventive healthcare decisions
- Reduce hospitalization costs
- Improve patient monitoring
- Assist doctors with clinical decision-making

---

## 📂 Dataset

**Dataset Name:** Stroke Prediction Dataset

The dataset contains medical and demographic information about patients.

### Features

| Feature | Description |
|----------|-------------|
| gender | Gender of patient |
| age | Age in years |
| hypertension | Whether patient has hypertension |
| heart_disease | Whether patient has heart disease |
| ever_married | Marital status |
| work_type | Type of employment |
| Residence_type | Urban or Rural residence |
| avg_glucose_level | Average glucose level |
| bmi | Body Mass Index |
| smoking_status | Smoking habit |
| stroke | Target variable (0 = No Stroke, 1 = Stroke) |

---

## 📊 Exploratory Data Analysis

The project includes multiple visualizations such as:

- Stroke distribution
- Age distribution
- Gender distribution
- Hypertension vs Stroke
- Heart Disease vs Stroke
- Smoking Status analysis
- BMI distribution
- Average Glucose Level distribution
- Correlation Heatmap
- Pair Plot
- Count Plots
- Box Plots
- Histograms
- Pie Charts

---

## ⚙️ Data Preprocessing

The following preprocessing steps were performed:

- Handling missing values
- Removing duplicate records
- Encoding categorical variables
- Feature engineering
- Feature scaling
- Train-Test Split
- Data validation

---

## 🤖 Machine Learning Models Used

The following classification algorithms were trained and evaluated:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

Hyperparameter tuning was performed using **GridSearchCV**.

Cross-validation was used to improve model reliability.

---

## 📈 Evaluation Metrics

Model performance was evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix

These metrics help evaluate the model's effectiveness in identifying stroke cases while minimizing false predictions.

---

## 🚀 Streamlit Application

A user-friendly Streamlit application was developed that allows users to:

- Enter patient information
- Predict stroke risk instantly
- Display prediction result
- Show prediction probability
- Provide an intuitive healthcare dashboard

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib

---

## 📁 Project Structure

```
Stroke-Prediction/
│
├── dataset/
│   └── healthcare-dataset-stroke-data.csv
│
├── notebook/
│   └── Stroke_Prediction.ipynb
│
├── models/
│   └── stroke_prediction_model.pkl
│
├── app.py
├── requirements.txt
├── README.md
└── assets/
```

---

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/stroke-prediction.git
```

Move into the project folder:

```bash
cd stroke-prediction
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## 📊 Workflow

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Engineering
5. Data Preprocessing
6. Model Training
7. Model Evaluation
8. Hyperparameter Tuning
9. Model Saving
10. Streamlit Deployment

---

## 💡 Key Insights

- Older individuals generally have a higher risk of stroke.
- Hypertension and heart disease significantly increase stroke risk.
- Elevated glucose levels are associated with higher stroke probability.
- Lifestyle factors such as smoking also influence stroke occurrence.
- Class imbalance should be considered while evaluating model performance.

---

## 📌 Future Improvements

- Deploy using Streamlit Cloud or Render
- Improve class imbalance handling using SMOTE
- Add Explainable AI (SHAP/LIME)
- Integrate with hospital management systems
- Enable real-time prediction using cloud APIs

---

## 👨‍💻 Author

**Pranav**

Machine Learning Project

---

## 📄 License

This project is developed for educational and academic purposes.
