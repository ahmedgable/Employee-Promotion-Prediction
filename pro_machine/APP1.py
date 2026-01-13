import streamlit as st
import pandas as pd
import pickle
import os

# -----------------------------
# Load models & preprocessors
# -----------------------------
model = pickle.load(open("catBoostingClassifier.pkl", "rb"))
scaler = pickle.load(open("standard_scaler.pkl", "rb"))
selector = pickle.load(open("selector.pkl", "rb"))

encoders = {}
encoder_cols = [
    "gender",
    "education_level",
    "job_role",
    "department",
    "company_type",
    "work_mode",
    "marital_status",
    "job_satisfaction"
]

for col in encoder_cols:
    encoders[col] = pickle.load(open(f"encoders/{col}_ohe.pkl", "rb"))

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🎯 Employee Promotion Prediction")

st.write("Enter employee information:")

input_data = {
    "gender": st.selectbox("Gender", ["Male", "Female"]),
    "education_level": st.selectbox("Education Level", ["Bachelor", "Master", "PhD"]),
    "job_role": st.text_input("Job Role"),
    "department": st.text_input("Department"),
    "company_type": st.selectbox("Company Type", ["Private", "Government"]),
    "work_mode": st.selectbox("Work Mode", ["Remote", "Hybrid", "Onsite"]),
    "marital_status": st.selectbox("Marital Status", ["Single", "Married"]),
    "job_satisfaction": st.selectbox("Job Satisfaction", ["Low", "Medium", "High"]),
}

df = pd.DataFrame([input_data])

# -----------------------------
# Encoding
# -----------------------------
for col, encoder in encoders.items():
    encoded = encoder.transform(df[[col]])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out([col])
    )
    df = pd.concat([df.drop(columns=[col]), encoded_df], axis=1)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Promotion"):
    df_scaled = scaler.transform(df)
    df_selected = selector.transform(df_scaled)

    prediction = model.predict(df_selected)[0]
    prob = model.predict_proba(df_selected)[0][1]

    if prediction == 1:
        st.success(f"✅ Promoted (Probability: {prob:.2f})")
    else:
        st.error(f"❌ Not Promoted (Probability: {prob:.2f})")
