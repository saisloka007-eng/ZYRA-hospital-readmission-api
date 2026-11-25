from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Hospital Readmission Prediction API")

# Load the 3-class model
model_data = joblib.load("readmission_model_improved.pkl")

model = model_data["model"]
feature_columns = model_data["feature_columns"]
classes = model_data["classes"]
threshold = model_data["threshold"]

IDX_LT30 = classes.index("<30")
IDX_GT30 = classes.index(">30")
IDX_NO = classes.index("NO")

LABEL_TEXT = {
    "NO": "Will NOT be readmitted",
    ">30": "Will be readmitted AFTER 30 days",
    "<30": "Will be readmitted WITHIN 30 days"
}


class PatientData(BaseModel):
    age: float
    time_in_hospital: int
    num_lab_procedures: int
    num_medications: int
    number_inpatient: int
    number_emergency: int
    number_outpatient: int
    admission_type_id: int


@app.get("/")
def home():
    return {
        "message": "Hospital Readmission Prediction API is running!"
    }


@app.post("/predict")
def predict(data: PatientData):

    # Convert patient input to DataFrame
    patient_data = data.model_dump()

    # Create model features
    df = pd.DataFrame([patient_data])

    df["age_mid"] = df["age"]
    df["meds_per_day"] = (
        df["num_medications"] / df["time_in_hospital"]
    )

    for t in range(1, 9):
        df[f"adm_{t}"] = (
            df["admission_type_id"] == t
        ).astype(int)

    # Select features in correct order
    input_df = df[feature_columns].astype(float)

    # Predict probabilities
    probabilities = model.predict_proba(input_df)[0]

    p_no = float(probabilities[IDX_NO])
    p_gt30 = float(probabilities[IDX_GT30])
    p_lt30 = float(probabilities[IDX_LT30])

    # Apply the same decision rule as training
    if p_lt30 >= threshold:
        prediction = "<30"
    else:
        prediction = "NO" if p_no >= p_gt30 else ">30"

    # Formatted readable output
    formatted_output = (
        f"Risk of readmission within 30 days : {p_lt30 * 100:.1f}%\n"
        f"Chance readmitted after 30 days    : {p_gt30 * 100:.1f}%\n"
        f"Chance NOT readmitted              : {p_no * 100:.1f}%\n"
        f"PREDICTION: {LABEL_TEXT[prediction]}\n"
        f"----------------------------------------------------"
    )

    return {
        "risk_within_30_days": round(p_lt30 * 100, 2),
        "chance_after_30_days": round(p_gt30 * 100, 2),
        "chance_not_readmitted": round(p_no * 100, 2),
        "threshold": round(float(threshold), 4),
        "predicted_class": prediction,
        "prediction": LABEL_TEXT[prediction],
        "formatted_output": formatted_output
    }