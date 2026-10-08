from fastapi import FastAPI
import joblib
from xgboost import XGBClassifier
from pydantic import BaseModel
import pandas as pd
import numpy as np




app = FastAPI()

model = XGBClassifier()
model.load_model("fraud_model.json")

scaler = joblib.load("scaler.pkl")

class Transaction(BaseModel):
    Time: float

    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    Amount: float


@app.get("/")
def home():
    return {
        "message": "Fraud Detection API is running!"
    }

@app.post("/predict")
def predict(transaction: Transaction):

    # Convert input to DataFrame
    data = transaction.model_dump()
    data = pd.DataFrame([data])

    # Create LogAmount
    data["LogAmount"] = np.log1p(data["Amount"])

    # Scale the same columns used during training
    columns_to_scale = ["Time", "Amount", "LogAmount"]

    data[columns_to_scale] = scaler.transform(
        data[columns_to_scale]
    )

    feature_names = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8",
    "V9", "V10", "V11", "V12", "V13", "V14", "V15",
    "V16", "V17", "V18", "V19", "V20", "V21", "V22",
    "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount",
    "LogAmount"
]
    data = data[feature_names]

   
    # Get fraud probability
    fraud_probability = model.predict_proba(data)[0,1]

    # Apply selected threshold
    threshold = 0.31
    prediction = int(fraud_probability >= threshold)

    return {
    "fraud_probability": float(fraud_probability),
    "threshold": threshold,
    "prediction": "Fraud" if prediction == 1 else "Legitimate"
    }
