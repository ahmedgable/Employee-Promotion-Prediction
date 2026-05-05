import streamlit as st
import pandas as pd
import pickle

# ====================== تحميل الموديلات ======================
@st.cache_resource
def load_artifacts():
    model = pickle.load(open("catBoostingClassifier.pkl", "rb"))
    scaler = pickle.load(open("standard_scaler.pkl", "rb"))
    selector = pickle.load(open("selector.pkl", "rb"))
    
    encoders = {}
    encoder_cols = ["gender", "education_level", "job_role", "department", 
                   "company_type", "work_mode", "marital_status", "job_satisfaction"]
    
    for col in encoder_cols:
        encoders[col] = pickle.load(open(f"encoders/{col}_ohe.pkl", "rb"))
    
    return model, scaler, selector, encoders

model, scaler, selector, encoders = load_artifacts()

# ====================== واجهة Streamlit ======================
st.title("🎯 Employee Promotion Prediction")
st.write("Enter employee information:")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    education = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])
    job_role = st.selectbox("Job Role", ["Engineer", "Manager", "Sales", "Designer", "HR"])
    department = st.selectbox("Department", ["IT", "Finance", "Marketing", "Operations", "HR"])
    company_type = st.selectbox("Company Type", ["Private", "SME", "Startup", "Government"])

with col2:
    work_mode = st.selectbox("Work Mode", ["Onsite", "Remote", "Hybrid"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married"])
    job_satisfaction = st.selectbox("Job Satisfaction", ["Low", "Medium", "High"])

# ====================== Numerical Features (كل الأعمدة الرقمية) ======================
st.subheader("📊 Numerical Features")

c1, c2, c3 = st.columns(3)

with c1:
    salary = st.number_input("Salary", 1000.0, 100000.0, 8000.0)
    experience = st.number_input("Experience Years", 0.0, 40.0, 5.0)
    working_hours = st.number_input("Working Hours/Week", 20, 80, 40)
    performance = st.number_input("Performance Score", 0.0, 100.0, 75.0)

with c2:
    projects = st.number_input("Projects Completed", 0, 50, 8)
    overtime = st.number_input("Overtime Hours", 0, 100, 10)
    absent_days = st.number_input("Absent Days", 0, 30, 3)
    training_hours = st.number_input("Training Hours", 0, 200, 20)

with c3:
    distance_from_home = st.number_input("Distance from Home (km)", 0.0, 100.0, 15.0)
    has_certifications = st.selectbox("Has Certifications", [1, 0])   # 1 = Yes, 0 = No
    # لو عندك promotion في التدريب بس مش محتاج تدخله

# ====================== Prediction ======================
if st.button("🔮 Predict Promotion", type="primary", use_container_width=True):
    
    input_data = {
        "salary": salary,
        "experience_years": experience,
        "working_hours_per_week": working_hours,
        "performance_score": performance,
        "projects_completed": projects,
        "overtime_hours": overtime,
        "distance_from_home_km": distance_from_home,
        "training_hours": training_hours,
        "absent_days": absent_days,
        "has_certifications": has_certifications,
        "gender": gender,
        "education_level": education,
        "job_role": job_role,
        "department": department,
        "company_type": company_type,
        "work_mode": work_mode,
        "marital_status": marital_status,
        "job_satisfaction": job_satisfaction,
    }

    df_input = pd.DataFrame([input_data])

    # --- Encoding ---
    for col, encoder in encoders.items():
        encoded = encoder.transform(df_input[[col]])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([col]))
        df_input = pd.concat([df_input.drop(columns=[col]), encoded_df], axis=1)

    # --- Scaling ---
    df_scaled = scaler.transform(df_input)          # ← هنا كان الإيرور

    # --- Feature Selection ---
    df_selected = selector.transform(df_scaled)

    # --- Prediction ---
    prediction = model.predict(df_selected)[0]
    probability = model.predict_proba(df_selected)[0][1]

    if prediction == 1:
        st.success(f"✅ Employee will be Promoted! (Probability: {probability:.1%})")
    else:
        st.error(f"❌ Employee will NOT be Promoted (Probability: {probability:.1%})")

    st.info(f"Raw Probability: {probability:.4f}")