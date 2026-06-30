from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# ==========================================================
# APP INIT
# ==========================================================
app = FastAPI(
    title="Multi Disease Prediction API",
    description="API for predicting Heart Disease, Stroke, and Diabetes risk",
    version="1.0.0"
)

# ==========================================================
# LOAD MODELS
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def load_pickle(file_name):
    path = os.path.join(MODELS_DIR, file_name)
    return joblib.load(path)


heart_model = load_pickle("heart_model.pkl")
heart_preprocessor = load_pickle("heart_preprocessor.pkl")

stroke_model = load_pickle("stroke_model.pkl")
stroke_preprocessor = load_pickle("stroke_preprocessor.pkl")

diabetes_model = load_pickle("diabetes_model.pkl")
diabetes_preprocessor = load_pickle("diabetes_preprocessor.pkl")


# ==========================================================
# REQUEST SCHEMAS (Pydantic)
# ==========================================================
class HeartInput(BaseModel):
    age: int
    sex: int          # 1 = Male, 0 = Female
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


class StrokeInput(BaseModel):
    gender: str        # "Male" / "Female" / "Other"
    age: float
    hypertension: int
    heart_disease: int
    ever_married: str  # "Yes" / "No"
    work_type: str      # "Private" / "Self-employed" / "Govt_job" / "children" / "Never_worked"
    Residence_type: str # "Urban" / "Rural"
    avg_glucose_level: float
    bmi: float
    smoking_status: str # "never smoked" / "formerly smoked" / "smokes" / "Unknown"


class DiabetesInput(BaseModel):
    gender: str          # "Male" / "Female"
    age: float
    hypertension: int
    heart_disease: int
    smoking_history: str # "never" / "current" / "former" / "ever" / "not current" / "No Info"
    bmi: float
    HbA1c_level: float
    blood_glucose_level: int


# ==========================================================
# ROOT
# ==========================================================
@app.get("/")
def read_root():
    return {
        "message": "Multi Disease Prediction API is running 🚀",
        "endpoints": ["/predict/heart", "/predict/stroke", "/predict/diabetes"],
        "docs": "/docs"
    }


# ==========================================================
# HEART DISEASE PREDICTION
# ==========================================================
@app.post("/predict/heart")
def predict_heart(data: HeartInput):
    try:
        input_df = pd.DataFrame([data.dict()])
        input_processed = heart_preprocessor.transform(input_df)

        prediction = heart_model.predict(input_processed)[0]
        probability = heart_model.predict_proba(input_processed)[0]

        # Dataset convention: target=0 -> Disease, target=1 -> No Disease
        risk_probability = round(float(probability[0]) * 100, 2)
        result = "Heart Disease Detected" if prediction == 0 else "No Heart Disease Detected"

        return {
            "prediction": int(prediction),
            "result": result,
            "risk_probability": risk_probability,
            "confidence": round(float(max(probability)) * 100, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==========================================================
# STROKE PREDICTION
# ==========================================================
@app.post("/predict/stroke")
def predict_stroke(data: StrokeInput):
    try:
        age = data.age
        bmi = data.bmi

        # Feature engineering (same as training)
        if age <= 30:
            age_group = "Young"
        elif age <= 50:
            age_group = "Middle"
        elif age <= 65:
            age_group = "Senior"
        else:
            age_group = "Old"

        if bmi < 18.5:
            bmi_category = "Underweight"
        elif bmi < 25:
            bmi_category = "Normal"
        elif bmi < 30:
            bmi_category = "Overweight"
        else:
            bmi_category = "Obese"

        input_df = pd.DataFrame([{
            "age": data.age,
            "avg_glucose_level": data.avg_glucose_level,
            "bmi": data.bmi,
            "gender": data.gender,
            "ever_married": data.ever_married,
            "work_type": data.work_type,
            "Residence_type": data.Residence_type,
            "smoking_status": data.smoking_status,
            "age_group": age_group,
            "bmi_category": bmi_category,
            "hypertension": data.hypertension,
            "heart_disease": data.heart_disease
        }])

        input_processed = stroke_preprocessor.transform(input_df)

        prediction = stroke_model.predict(input_processed)[0]
        probability = stroke_model.predict_proba(input_processed)[0]

        risk_probability = round(float(probability[1]) * 100, 2)
        result = "High Risk of Stroke" if prediction == 1 else "Low Risk of Stroke"

        return {
            "prediction": int(prediction),
            "result": result,
            "risk_probability": risk_probability,
            "confidence": round(float(max(probability)) * 100, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==========================================================
# DIABETES PREDICTION
# ==========================================================
@app.post("/predict/diabetes")
def predict_diabetes(data: DiabetesInput):
    try:
        input_df = pd.DataFrame([data.dict()])
        input_processed = diabetes_preprocessor.transform(input_df)

        prediction = diabetes_model.predict(input_processed)[0]
        probability = diabetes_model.predict_proba(input_processed)[0]

        risk_probability = round(float(probability[1]) * 100, 2)
        result = "High Risk of Diabetes" if prediction == 1 else "Low Risk of Diabetes"

        return {
            "prediction": int(prediction),
            "result": result,
            "risk_probability": risk_probability,
            "confidence": round(float(max(probability)) * 100, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))