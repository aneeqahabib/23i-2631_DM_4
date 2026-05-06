import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

model = joblib.load("heart_rf_model.pkl")
feature_names = joblib.load("feature_names.pkl")

st.set_page_config(page_title="Heart Disease Dashboard")

st.title("Heart Disease Prediction Dashboard")

st.write("Enter patient information and click Predict.")

# ------------------------------------------------------------
# DEFAULT TEST PATIENT
# ------------------------------------------------------------

default_patient = {
    "age": 63,
    "sex": 1,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "ca": 0,

    "cp_1.0": 1,
    "cp_2.0": 0,
    "cp_3.0": 0,
    "cp_4.0": 0,

    "restecg_0.0": 0,
    "restecg_1.0": 0,
    "restecg_2.0": 1,

    "slope_1.0": 0,
    "slope_2.0": 0,
    "slope_3.0": 1,

    "thal_3.0": 0,
    "thal_6.0": 1,
    "thal_7.0": 0
}

# ------------------------------------------------------------
# INPUT FORM
# ------------------------------------------------------------

st.subheader("Patient Input")

age = st.number_input("Age (20–80)", 20, 80, default_patient["age"])
sex = st.selectbox("Sex (0=female, 1=male)", [0, 1], index=default_patient["sex"])
trestbps = st.number_input("Resting BP (80–220)", 80, 220, default_patient["trestbps"])
chol = st.number_input("Cholesterol (100–600)", 100, 600, default_patient["chol"])
fbs = st.selectbox("Fasting Blood Sugar > 120 (0/1)", [0, 1], index=default_patient["fbs"])
thalach = st.number_input("Max Heart Rate (70–210)", 70, 210, default_patient["thalach"])
exang = st.selectbox("Exercise Induced Angina (0/1)", [0, 1], index=default_patient["exang"])
oldpeak = st.number_input("ST Depression (0–7)", 0.0, 7.0, float(default_patient["oldpeak"]))
ca = st.number_input("Number of Major Vessels (0–4)", 0, 4, default_patient["ca"])

cp = st.selectbox("Chest Pain Type", [1.0, 2.0, 3.0, 4.0])
restecg = st.selectbox("Rest ECG", [0.0, 1.0, 2.0])
slope = st.selectbox("Slope", [1.0, 2.0, 3.0])
thal = st.selectbox("Thal", [3.0, 6.0, 7.0])

# ------------------------------------------------------------
# BUILD INPUT ROW
# ------------------------------------------------------------

def build_input():
    row = {col: 0 for col in feature_names}

    for col in row:
        if col == "age":
            row[col] = age
        elif col == "sex":
            row[col] = sex
        elif col == "trestbps":
            row[col] = trestbps
        elif col == "chol":
            row[col] = chol
        elif col == "fbs":
            row[col] = fbs
        elif col == "thalach":
            row[col] = thalach
        elif col == "exang":
            row[col] = exang
        elif col == "oldpeak":
            row[col] = oldpeak
        elif col == "ca":
            row[col] = ca

    cp_col = f"cp_{cp}"
    rest_col = f"restecg_{restecg}"
    slope_col = f"slope_{slope}"
    thal_col = f"thal_{thal}"

    if cp_col in row:
        row[cp_col] = 1
    if rest_col in row:
        row[rest_col] = 1
    if slope_col in row:
        row[slope_col] = 1
    if thal_col in row:
        row[thal_col] = 1

    return pd.DataFrame([row])

# ------------------------------------------------------------
# PREDICT
# ------------------------------------------------------------

if st.button("Predict"):

    X_input = build_input()

    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][1]

    st.subheader("Prediction Result")

    if pred == 0:
        st.markdown("### 🟢 No Disease")
    else:
        st.markdown("### 🔴 Disease Present")

    st.write(f"Confidence: **{prob*100:.2f}%**")

    # --------------------------------------------------------
    # TOP 3 FEATURES
    # --------------------------------------------------------

    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    top3 = importance_df.head(3)

    st.subheader("Top 3 Important Features")

    fig, ax = plt.subplots()
    ax.barh(top3["feature"], top3["importance"])
    ax.invert_yaxis()
    st.pyplot(fig)

    # --------------------------------------------------------
    # NURSE-FRIENDLY EXPLANATION
    # --------------------------------------------------------

    st.subheader("Clinical Note")

    top_names = ", ".join(top3["feature"].tolist())

    if pred == 1:
        st.write(
            f"This patient shows elevated cardiac risk. "
            f"The strongest contributing factors are {top_names}. "
            f"Further clinical evaluation is recommended."
        )
    else:
        st.write(
            f"This patient shows relatively lower cardiac risk. "
            f"The most influential factors were {top_names}. "
            f"Routine monitoring is still recommended."
        )