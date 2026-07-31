from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
import os

# ==========================================================
# APP INIT
# ==========================================================
app = FastAPI(
    title="Multi Disease Prediction API",
    description="API for predicting Heart Disease, Stroke, Diabetes and Breast Cancer risk",
    version="2.0.0"
)

# ==========================================================
# LOAD MODELS
# ==========================================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

def load_pickle(file_name):
    path = os.path.join(MODELS_DIR, file_name)
    return joblib.load(path)

heart_model        = load_pickle("heart_model.pkl")
heart_preprocessor = load_pickle("heart_preprocessor.pkl")

stroke_model        = load_pickle("stroke_model.pkl")
stroke_preprocessor = load_pickle("stroke_preprocessor.pkl")

diabetes_model        = load_pickle("diabetes_model.pkl")
diabetes_preprocessor = load_pickle("diabetes_preprocessor.pkl")

breast_cancer_model        = load_pickle("breast_cancer_model.pkl")
breast_cancer_preprocessor = load_pickle("breast_cancer_preprocessor.pkl")

# ==========================================================
# REQUEST SCHEMAS (Pydantic)
# ==========================================================
class HeartInput(BaseModel):
    age: int
    sex: int
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
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    ever_married: str
    work_type: str
    Residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str


class DiabetesInput(BaseModel):
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    smoking_history: str
    bmi: float
    HbA1c_level: float
    blood_glucose_level: int


class BreastCancerInput(BaseModel):
    radius_mean: float
    texture_mean: float
    perimeter_mean: float
    area_mean: float
    smoothness_mean: float
    compactness_mean: float
    concavity_mean: float
    concave_points_mean: float
    symmetry_mean: float
    fractal_dimension_mean: float
    radius_se: float
    texture_se: float
    perimeter_se: float
    area_se: float
    smoothness_se: float
    compactness_se: float
    concavity_se: float
    concave_points_se: float
    symmetry_se: float
    fractal_dimension_se: float
    radius_worst: float
    texture_worst: float
    perimeter_worst: float
    area_worst: float
    smoothness_worst: float
    compactness_worst: float
    concavity_worst: float
    concave_points_worst: float
    symmetry_worst: float
    fractal_dimension_worst: float


# ==========================================================
# ROOT
# ==========================================================
@app.get("/")
def read_root():
    return {
        "message": "Multi Disease Prediction API is running 🚀",
        "endpoints": [
            "/predict/heart",
            "/predict/stroke",
            "/predict/diabetes",
            "/predict/breast-cancer"
        ],
        "docs": "/docs"
    }


# ==========================================================
# HEART DISEASE
# ==========================================================
@app.post("/predict/heart")
def predict_heart(data: HeartInput):
    try:
        input_df        = pd.DataFrame([data.dict()])
        input_processed = heart_preprocessor.transform(input_df)
        prediction      = heart_model.predict(input_processed)[0]
        probability     = heart_model.predict_proba(input_processed)[0]

        risk_probability = round(float(probability[1]) * 100, 2)
        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease Detected"

        return {
            "prediction": int(prediction),
            "result": result,
            "risk_probability": risk_probability,
            "confidence": round(float(max(probability)) * 100, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==========================================================
# STROKE
# ==========================================================
@app.post("/predict/stroke")
def predict_stroke(data: StrokeInput):
    try:
        age = data.age
        bmi = data.bmi

        if age <= 30:   age_group = "Young"
        elif age <= 50: age_group = "Middle"
        elif age <= 65: age_group = "Senior"
        else:           age_group = "Old"

        if bmi < 18.5:  bmi_category = "Underweight"
        elif bmi < 25:  bmi_category = "Normal"
        elif bmi < 30:  bmi_category = "Overweight"
        else:           bmi_category = "Obese"

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

        input_processed  = stroke_preprocessor.transform(input_df)
        prediction       = stroke_model.predict(input_processed)[0]
        probability      = stroke_model.predict_proba(input_processed)[0]

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
# DIABETES
# ==========================================================
@app.post("/predict/diabetes")
def predict_diabetes(data: DiabetesInput):
    try:
        input_df        = pd.DataFrame([data.dict()])
        input_processed = diabetes_preprocessor.transform(input_df)
        prediction      = diabetes_model.predict(input_processed)[0]
        probability     = diabetes_model.predict_proba(input_processed)[0]

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


# ==========================================================
# BREAST CANCER
# ==========================================================
@app.post("/predict/breast-cancer")
def predict_breast_cancer(data: BreastCancerInput):
    try:
        input_df = pd.DataFrame([{
            "radius_mean": data.radius_mean,
            "texture_mean": data.texture_mean,
            "perimeter_mean": data.perimeter_mean,
            "area_mean": data.area_mean,
            "smoothness_mean": data.smoothness_mean,
            "compactness_mean": data.compactness_mean,
            "concavity_mean": data.concavity_mean,
            "concave points_mean": data.concave_points_mean,
            "symmetry_mean": data.symmetry_mean,
            "fractal_dimension_mean": data.fractal_dimension_mean,
            "radius_se": data.radius_se,
            "texture_se": data.texture_se,
            "perimeter_se": data.perimeter_se,
            "area_se": data.area_se,
            "smoothness_se": data.smoothness_se,
            "compactness_se": data.compactness_se,
            "concavity_se": data.concavity_se,
            "concave points_se": data.concave_points_se,
            "symmetry_se": data.symmetry_se,
            "fractal_dimension_se": data.fractal_dimension_se,
            "radius_worst": data.radius_worst,
            "texture_worst": data.texture_worst,
            "perimeter_worst": data.perimeter_worst,
            "area_worst": data.area_worst,
            "smoothness_worst": data.smoothness_worst,
            "compactness_worst": data.compactness_worst,
            "concavity_worst": data.concavity_worst,
            "concave points_worst": data.concave_points_worst,
            "symmetry_worst": data.symmetry_worst,
            "fractal_dimension_worst": data.fractal_dimension_worst,
        }])

        input_processed  = breast_cancer_preprocessor.transform(input_df)
        prediction       = breast_cancer_model.predict(input_processed)[0]
        probability      = breast_cancer_model.predict_proba(input_processed)[0]

        risk_probability = round(float(probability[1]) * 100, 2)
        result = "Malignant (Cancerous)" if prediction == 1 else "Benign (Non-Cancerous)"

        return {
            "prediction": int(prediction),
            "result": result,
            "risk_probability": risk_probability,
            "confidence": round(float(max(probability)) * 100, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))