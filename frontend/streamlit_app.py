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
API_URL = "http://127.0.0.1:8000"   # Local FastAPI. Deploy hone ke baad isse update karna.

# ==========================================================
# CUSTOM CSS
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
.danger-box { background: linear-gradient(135deg, #7F1D1D, #991B1B); color: white; border: 1px solid #EF4444; }
.metric-card { background: #161B22; border: 1px solid #1F2937; border-radius: 12px; padding: 16px; text-align: center; }
hr { border-color: #1F2937; }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# SIDEBAR
# ==========================================================
st.sidebar.title("🏥 Disease Prediction")
page = st.sidebar.radio("Choose Disease", ["🏠 Home", "🫀 Heart Disease", "🧠 Stroke", "🩸 Diabetes"])
st.sidebar.markdown("---")
st.sidebar.info("This app predicts ✅ Heart Disease ✅ Stroke ✅ Diabetes via a FastAPI backend.")
st.sidebar.markdown("---")
st.sidebar.warning("⚠️ Educational purposes only. Always consult a doctor.")

# ==========================================================
# HOME PAGE
# ==========================================================
if page == "🏠 Home":
    st.title("🏥 Multi Disease Prediction System")
    st.write("AI-powered risk prediction — Streamlit frontend + FastAPI backend architecture.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card"><h3>🫀</h3><b>Heart Disease</b><br>Logistic Regression</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h3>🧠</h3><b>Stroke</b><br>Logistic Regression</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h3>🩸</h3><b>Diabetes</b><br>XGBoost</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    ### Architecture
    ```
    Streamlit (UI)  →  FastAPI (/predict/...)  →  ML Model  →  Response
    ```
    Make sure the FastAPI backend (`uvicorn main:app --reload`) is running before using this app.
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
        age = st.number_input("Age", 1, 120, 45)
        sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
        trestbps = st.number_input("Resting Blood Pressure", 50, 250, 120)
        chol = st.number_input("Serum Cholesterol", 100, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
        restecg = st.selectbox("Resting ECG", [0, 1, 2])
    with col2:
        thalach = st.number_input("Maximum Heart Rate Achieved", 50, 250, 150)
        exang = st.selectbox("Exercise Induced Angina", [0, 1])
        oldpeak = st.number_input("Old Peak", 0.0, 10.0, 1.0, step=0.1)
        slope = st.selectbox("Slope", [0, 1, 2])
        ca = st.selectbox("Major Vessels (0-4)", [0, 1, 2, 3, 4])
        thal = st.selectbox("Thal", [0, 1, 2, 3])

    st.markdown("---")
    if st.button("❤️ Predict Heart Disease"):
        payload = {
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
            "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
            "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
        }
        try:
            response = requests.post(f"{API_URL}/predict/heart", json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()

            st.markdown("---")
            if result["prediction"] == 0:
                st.markdown(f"""
                <div class="pred-box danger-box">❤️ {result['result']}<br><br>
                <h2>{result['risk_probability']}% Risk</h2></div>
                """, unsafe_allow_html=True)
                st.error("⚠️ HIGH risk of Heart Disease. Please consult a cardiologist.")
            else:
                st.markdown(f"""
                <div class="pred-box success-box">💚 {result['result']}<br><br>
                <h2>{100 - result['risk_probability']:.2f}% Safe</h2></div>
                """, unsafe_allow_html=True)
                st.success("✅ LOW risk of Heart Disease.")

            st.progress(result["risk_probability"] / 100)
            st.info(f"Model Confidence: {result['confidence']}%")

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
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        age = st.number_input("Age", 1, 100, 45)
        hypertension = st.selectbox("Hypertension", [0, 1])
        heart_disease = st.selectbox("Heart Disease", [0, 1])
        ever_married = st.selectbox("Ever Married", ["Yes", "No"])
        work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    with col2:
        residence = st.selectbox("Residence Type", ["Urban", "Rural"])
        glucose = st.number_input("Average Glucose Level", 50.0, 300.0, 100.0)
        bmi = st.number_input("BMI", 10.0, 100.0, 25.0)
        smoking = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

    st.markdown("---")
    if st.button("🧠 Predict Stroke"):
        payload = {
            "gender": gender, "age": age, "hypertension": hypertension,
            "heart_disease": heart_disease, "ever_married": ever_married,
            "work_type": work_type, "Residence_type": residence,
            "avg_glucose_level": glucose, "bmi": bmi, "smoking_status": smoking
        }
        try:
            response = requests.post(f"{API_URL}/predict/stroke", json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()

            st.markdown("---")
            if result["prediction"] == 1:
                st.markdown(f"""
                <div class="pred-box danger-box">🧠 {result['result']}<br><br>
                <h2>{result['risk_probability']}% Risk</h2></div>
                """, unsafe_allow_html=True)
                st.error("⚠️ High probability of Stroke. Please consult a neurologist.")
            else:
                st.markdown(f"""
                <div class="pred-box success-box">💚 {result['result']}<br><br>
                <h2>{100 - result['risk_probability']:.2f}% Safe</h2></div>
                """, unsafe_allow_html=True)
                st.success("✅ LOW risk of Stroke.")

            st.progress(result["risk_probability"] / 100)
            st.info(f"Model Confidence: {result['confidence']}%")

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
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", 1, 120, 40)
        hypertension = st.selectbox("Hypertension", [0, 1])
        heart_disease = st.selectbox("Heart Disease", [0, 1])
    with col2:
        smoking_history = st.selectbox("Smoking History", ["never", "current", "former", "ever", "not current", "No Info"])
        bmi = st.number_input("BMI", 10.0, 80.0, 25.0, step=0.1)
        hba1c = st.number_input("HbA1c Level", 3.0, 15.0, 5.5, step=0.1)
        glucose = st.number_input("Blood Glucose Level", 50, 400, 120)

    st.markdown("---")
    if st.button("🩸 Predict Diabetes"):
        payload = {
            "gender": gender, "age": age, "hypertension": hypertension,
            "heart_disease": heart_disease, "smoking_history": smoking_history,
            "bmi": bmi, "HbA1c_level": hba1c, "blood_glucose_level": glucose
        }
        try:
            response = requests.post(f"{API_URL}/predict/diabetes", json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()

            st.markdown("---")
            if result["prediction"] == 1:
                st.markdown(f"""
                <div class="pred-box danger-box">🩸 {result['result']}<br><br>
                <h2>{result['risk_probability']}% Risk</h2></div>
                """, unsafe_allow_html=True)
                st.error("⚠️ HIGH risk of Diabetes. Please consult a physician.")
            else:
                st.markdown(f"""
                <div class="pred-box success-box">💚 {result['result']}<br><br>
                <h2>{100 - result['risk_probability']:.2f}% Safe</h2></div>
                """, unsafe_allow_html=True)
                st.success("✅ LOW risk of Diabetes.")

            st.progress(result["risk_probability"] / 100)
            st.info(f"Model Confidence: {result['confidence']}%")

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
<p>Heart Disease • Stroke • Diabetes</p>
<p style="font-size:13px; color:gray;">Powered by Streamlit + FastAPI</p>
</div>
""", unsafe_allow_html=True)