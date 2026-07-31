import streamlit as st
import requests

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="Multi Disease Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# CONFIG — FastAPI Backend URL
# ==========================================================
API_URL = "http://127.0.0.1:8000"

# ==========================================================
# CUSTOM CSS  (same UI as before)
# ==========================================================
st.markdown("""
<style>
.stApp { background: linear-gradient(180deg, #0E1117 0%, #131722 100%); }
h1, h2, h3, h4 { color: #F1F5F9; font-family: 'Segoe UI', sans-serif; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; }
[data-testid="stSidebar"] { background: #111827; border-right: 1px solid #1F2937; }
.stButton > button {
    width: 100%; background: linear-gradient(90deg, #2563EB, #1D4ED8);
    color: white; border-radius: 10px; border: none; height: 3em;
    font-size: 17px; font-weight: 600; transition: 0.2s ease-in-out;
}
.stButton > button:hover { background: linear-gradient(90deg, #1D4ED8, #1E40AF); transform: translateY(-1px); }
.pred-box { padding: 24px; border-radius: 14px; font-size: 22px; font-weight: bold; text-align: center; margin-bottom: 10px; }
.success-box { background: linear-gradient(135deg, #064E3B, #065F46); color: white; border: 1px solid #10B981; }
.danger-box  { background: linear-gradient(135deg, #7F1D1D, #991B1B); color: white; border: 1px solid #EF4444; }
.metric-card { background: #161B22; border: 1px solid #1F2937; border-radius: 12px; padding: 16px; text-align: center; }
hr { border-color: #1F2937; }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# SIDEBAR
# ==========================================================
st.sidebar.title("🏥 Disease Prediction")
page = st.sidebar.radio(
    "Choose Disease",
    ["🏠 Home", "🫀 Heart Disease", "🧠 Stroke", "🩸 Diabetes", "🎗️ Breast Cancer"]
)
st.sidebar.markdown("---")
st.sidebar.info("This app predicts ✅ Heart Disease ✅ Stroke ✅ Diabetes ✅ Breast Cancer via a FastAPI backend.")
st.sidebar.markdown("---")
st.sidebar.warning("⚠️ Educational purposes only. Always consult a doctor.")


# ==========================================================
# HELPER — show result block
# ==========================================================
def show_result(prediction, result_text, risk_probability, confidence, positive_val=1):
    st.markdown("---")
    if prediction == positive_val:
        st.markdown(f"""
        <div class="pred-box danger-box">⚠️ {result_text}<br><br>
        <h2>{risk_probability}% Risk</h2></div>
        """, unsafe_allow_html=True)
        st.error("⚠️ Please consult a healthcare professional immediately.")
    else:
        st.markdown(f"""
        <div class="pred-box success-box">💚 {result_text}<br><br>
        <h2>{100 - risk_probability:.2f}% Safe</h2></div>
        """, unsafe_allow_html=True)
        st.success("✅ Low risk detected. Maintain regular health checkups.")

    st.progress(risk_probability / 100)
    st.info(f"Model Confidence: {confidence}%")


# ==========================================================
# HOME
# ==========================================================
if page == "🏠 Home":
    st.title("🏥 Multi Disease Prediction System")
    st.write("AI-powered risk prediction — Streamlit frontend + FastAPI backend architecture.")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><h3>🫀</h3><b>Heart Disease</b><br>Logistic Regression</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h3>🧠</h3><b>Stroke</b><br>Logistic Regression</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h3>🩸</h3><b>Diabetes</b><br>XGBoost</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><h3>🎗️</h3><b>Breast Cancer</b><br>Logistic Regression</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
### 🏗️ Architecture

- **Frontend:** Streamlit
- **Backend:** FastAPI
- **Model:** Machine Learning
- **Output:** Prediction Response

> **Note:** Ensure the FastAPI backend is running before using this application.
""")


# ==========================================================
# HEART DISEASE
# ==========================================================
elif page == "🫀 Heart Disease":
    st.title("🫀 Heart Disease Prediction")
    st.write("Fill all the details below to predict the possibility of Heart Disease.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        age      = st.number_input("Age", 1, 120, 45)
        sex      = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
        cp       = st.selectbox("Chest Pain Type", [0, 1, 2, 3],
                                format_func=lambda x: {0:"Typical Angina", 1:"Atypical Angina",
                                                        2:"Non-Anginal Pain", 3:"Asymptomatic"}[x])
        trestbps = st.number_input("Resting Blood Pressure", 50, 250, 120)
        chol     = st.number_input("Serum Cholesterol", 100, 600, 200)
        fbs      = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1],
                                format_func=lambda x: "Yes" if x == 1 else "No")
        restecg  = st.selectbox("Resting ECG", [0, 1, 2],
                                format_func=lambda x: {0:"Normal", 1:"ST-T Wave Abnormality",
                                                        2:"Left Ventricular Hypertrophy"}[x])
    with col2:
        thalach = st.number_input("Maximum Heart Rate Achieved", 50, 250, 150)
        exang   = st.selectbox("Exercise Induced Angina", [0, 1],
                               format_func=lambda x: "Yes" if x == 1 else "No")
        oldpeak = st.number_input("Old Peak", 0.0, 10.0, 1.0, step=0.1)
        slope   = st.selectbox("Slope of Peak Exercise ST", [0, 1, 2],
                               format_func=lambda x: {0:"Upsloping", 1:"Flat", 2:"Downsloping"}[x])
        ca      = st.selectbox("Major Vessels (0-4)", [0, 1, 2, 3, 4])
        thal    = st.selectbox("Thalassemia", [0, 1, 2, 3],
                               format_func=lambda x: {0:"Unknown", 1:"Normal",
                                                       2:"Fixed Defect", 3:"Reversible Defect"}[x])

    st.markdown("---")
    if st.button("❤️ Predict Heart Disease"):
        payload = {
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
            "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
            "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
        }
        try:
            r = requests.post(f"{API_URL}/predict/heart", json=payload, timeout=10)
            r.raise_for_status()
            res = r.json()
            show_result(res["prediction"], res["result"], res["risk_probability"], res["confidence"])
        except requests.exceptions.RequestException as e:
            st.error("⚠️ Could not reach FastAPI backend. Is it running?")
            st.exception(e)


# ==========================================================
# STROKE
# ==========================================================
elif page == "🧠 Stroke":
    st.title("🧠 Stroke Prediction")
    st.write("Fill all patient information to predict Stroke risk.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        gender        = st.selectbox("Gender", ["Male", "Female", "Other"])
        age           = st.number_input("Age", 1, 100, 45)
        hypertension  = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        ever_married  = st.selectbox("Ever Married", ["Yes", "No"])
        work_type     = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    with col2:
        residence = st.selectbox("Residence Type", ["Urban", "Rural"])
        glucose   = st.number_input("Average Glucose Level", 50.0, 300.0, 100.0)
        bmi       = st.number_input("BMI", 10.0, 100.0, 25.0)
        smoking   = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

    st.markdown("---")
    if st.button("🧠 Predict Stroke"):
        payload = {
            "gender": gender, "age": age, "hypertension": hypertension,
            "heart_disease": heart_disease, "ever_married": ever_married,
            "work_type": work_type, "Residence_type": residence,
            "avg_glucose_level": glucose, "bmi": bmi, "smoking_status": smoking
        }
        try:
            r = requests.post(f"{API_URL}/predict/stroke", json=payload, timeout=10)
            r.raise_for_status()
            res = r.json()
            show_result(res["prediction"], res["result"], res["risk_probability"], res["confidence"])
        except requests.exceptions.RequestException as e:
            st.error("⚠️ Could not reach FastAPI backend. Is it running?")
            st.exception(e)


# ==========================================================
# DIABETES
# ==========================================================
elif page == "🩸 Diabetes":
    st.title("🩸 Diabetes Prediction")
    st.write("Fill all patient information to predict Diabetes risk.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        gender        = st.selectbox("Gender", ["Male", "Female"])
        age           = st.number_input("Age", 1, 120, 40)
        hypertension  = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    with col2:
        smoking_history = st.selectbox("Smoking History", ["never", "current", "former", "ever", "not current", "No Info"])
        bmi             = st.number_input("BMI", 10.0, 80.0, 25.0, step=0.1)
        hba1c           = st.number_input("HbA1c Level", 3.0, 15.0, 5.5, step=0.1)
        glucose         = st.number_input("Blood Glucose Level", 50, 400, 120)

    st.markdown("---")
    if st.button("🩸 Predict Diabetes"):
        payload = {
            "gender": gender, "age": age, "hypertension": hypertension,
            "heart_disease": heart_disease, "smoking_history": smoking_history,
            "bmi": bmi, "HbA1c_level": hba1c, "blood_glucose_level": glucose
        }
        try:
            r = requests.post(f"{API_URL}/predict/diabetes", json=payload, timeout=10)
            r.raise_for_status()
            res = r.json()
            show_result(res["prediction"], res["result"], res["risk_probability"], res["confidence"])
        except requests.exceptions.RequestException as e:
            st.error("⚠️ Could not reach FastAPI backend. Is it running?")
            st.exception(e)


# ==========================================================
# BREAST CANCER
# ==========================================================
elif page == "🎗️ Breast Cancer":
    st.title("🎗️ Breast Cancer Classification")
    st.write("Enter cell nucleus measurements from a Fine Needle Aspirate (FNA) biopsy.")
    st.markdown("---")
    st.info("💡 Three groups: **Mean** = average nucleus · **SE** = variability · **Worst** = largest nucleus in sample")

    # ── MEAN FEATURES ──────────────────────────────────────
    st.subheader("📊 Mean Features")
    col1, col2, col3 = st.columns(3)
    with col1:
        radius_mean    = st.number_input("Radius Mean",    1.0,  35.0,  14.1,  0.01)
        texture_mean   = st.number_input("Texture Mean",   1.0,  45.0,  19.3,  0.01)
    with col2:
        perimeter_mean = st.number_input("Perimeter Mean", 40.0, 200.0, 92.0,  0.1)
        area_mean      = st.number_input("Area Mean",      100.0,2600.0, 655.0, 1.0)
    with col3:
        smoothness_mean  = st.number_input("Smoothness Mean",  0.05, 0.17, 0.096, 0.001)
        compactness_mean = st.number_input("Compactness Mean", 0.01, 0.40, 0.104, 0.001)

    col4, col5, col6 = st.columns(3)
    with col4: concavity_mean      = st.number_input("Concavity Mean",      0.0,  0.45, 0.089, 0.001)
    with col5: concave_points_mean = st.number_input("Concave Points Mean", 0.0,  0.21, 0.049, 0.001)
    with col6: symmetry_mean       = st.number_input("Symmetry Mean",       0.10, 0.31, 0.181, 0.001)

    fractal_dimension_mean = st.number_input("Fractal Dimension Mean", 0.04, 0.10, 0.063, 0.001)

    st.markdown("---")

    # ── SE FEATURES ────────────────────────────────────────
    st.subheader("📏 Standard Error (SE) Features")
    col7, col8, col9 = st.columns(3)
    with col7:
        radius_se  = st.number_input("Radius SE",    0.1,  3.0,   0.405, 0.001)
        texture_se = st.number_input("Texture SE",   0.3,  5.0,   1.217, 0.001)
    with col8:
        perimeter_se = st.number_input("Perimeter SE", 0.7,  22.0,  2.866, 0.01)
        area_se      = st.number_input("Area SE",      6.0,  550.0, 40.3,  0.1)
    with col9:
        smoothness_se  = st.number_input("Smoothness SE",  0.001, 0.032, 0.007, 0.0001, format="%.4f")
        compactness_se = st.number_input("Compactness SE", 0.002, 0.135, 0.025, 0.001)

    col10, col11, col12 = st.columns(3)
    with col10: concavity_se      = st.number_input("Concavity SE",      0.0,  0.40, 0.032, 0.001)
    with col11: concave_points_se = st.number_input("Concave Points SE", 0.0,  0.05, 0.012, 0.001)
    with col12: symmetry_se       = st.number_input("Symmetry SE",       0.007,0.08, 0.020, 0.001)

    fractal_dimension_se = st.number_input("Fractal Dimension SE", 0.0008, 0.030, 0.004, 0.0001, format="%.4f")

    st.markdown("---")

    # ── WORST FEATURES ─────────────────────────────────────
    st.subheader("⚠️ Worst (Largest) Features")
    col13, col14, col15 = st.columns(3)
    with col13:
        radius_worst  = st.number_input("Radius Worst",    7.0,  37.0,   16.3,  0.01)
        texture_worst = st.number_input("Texture Worst",   12.0, 50.0,   25.7,  0.01)
    with col14:
        perimeter_worst = st.number_input("Perimeter Worst", 50.0, 252.0, 107.3, 0.1)
        area_worst      = st.number_input("Area Worst",      185.0,4255.0, 880.6, 1.0)
    with col15:
        smoothness_worst  = st.number_input("Smoothness Worst",  0.07, 0.22, 0.132, 0.001)
        compactness_worst = st.number_input("Compactness Worst", 0.02, 1.10, 0.254, 0.001)

    col16, col17, col18 = st.columns(3)
    with col16: concavity_worst      = st.number_input("Concavity Worst",      0.0,  1.25, 0.272, 0.001)
    with col17: concave_points_worst = st.number_input("Concave Points Worst", 0.0,  0.30, 0.115, 0.001)
    with col18: symmetry_worst       = st.number_input("Symmetry Worst",       0.15, 0.66, 0.290, 0.001)

    fractal_dimension_worst = st.number_input("Fractal Dimension Worst", 0.05, 0.21, 0.084, 0.001)

    st.markdown("---")
    if st.button("🎗️ Classify Tumour"):
        payload = {
            "radius_mean": radius_mean, "texture_mean": texture_mean,
            "perimeter_mean": perimeter_mean, "area_mean": area_mean,
            "smoothness_mean": smoothness_mean, "compactness_mean": compactness_mean,
            "concavity_mean": concavity_mean, "concave_points_mean": concave_points_mean,
            "symmetry_mean": symmetry_mean, "fractal_dimension_mean": fractal_dimension_mean,
            "radius_se": radius_se, "texture_se": texture_se,
            "perimeter_se": perimeter_se, "area_se": area_se,
            "smoothness_se": smoothness_se, "compactness_se": compactness_se,
            "concavity_se": concavity_se, "concave_points_se": concave_points_se,
            "symmetry_se": symmetry_se, "fractal_dimension_se": fractal_dimension_se,
            "radius_worst": radius_worst, "texture_worst": texture_worst,
            "perimeter_worst": perimeter_worst, "area_worst": area_worst,
            "smoothness_worst": smoothness_worst, "compactness_worst": compactness_worst,
            "concavity_worst": concavity_worst, "concave_points_worst": concave_points_worst,
            "symmetry_worst": symmetry_worst, "fractal_dimension_worst": fractal_dimension_worst,
        }
        try:
            r = requests.post(f"{API_URL}/predict/breast-cancer", json=payload, timeout=10)
            r.raise_for_status()
            res = r.json()
            show_result(res["prediction"], res["result"], res["risk_probability"], res["confidence"])
        except requests.exceptions.RequestException as e:
            st.error("⚠️ Could not reach FastAPI backend. Is it running?")
            st.exception(e)


# ==========================================================
# FOOTER
# ==========================================================
st.markdown("---")
st.markdown("""
<div style="text-align:center">
<h3>🏥 Multi Disease Prediction System</h3>
<p>Heart Disease • Stroke • Diabetes • Breast Cancer</p>
<p style="font-size:13px; color:gray;">Powered by Streamlit + FastAPI</p>
</div>
""", unsafe_allow_html=True)