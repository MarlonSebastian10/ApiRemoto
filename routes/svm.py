from fastapi import APIRouter
import joblib
import numpy as np
from schemas import SoilData

router = APIRouter()

svm_model = joblib.load("models/svm_model.pkl")
scaler = joblib.load("models/scaler.pkl")

@router.post("/predict/svm")
def predict_svm(data: SoilData):
    input_data = np.array([[data.nitrogeno, data.fosforo, data.potasio, 
                            data.temperatura, data.humedad, data.ph, data.lluvia]])
    
    input_scaled = scaler.transform(input_data)
    prediction = svm_model.predict(input_scaled)
    return {"model": "SVM", "Cultivo Recomendado": prediction[0]}
