from fastapi import APIRouter
import joblib
import numpy as np
from schemas import SoilData

router = APIRouter()

rf_model = joblib.load("models/random_forest_model.pkl")
scaler = joblib.load("models/scaler.pkl")

@router.post("/predict/random_forest")
def predict_rf(data: SoilData):
    input_data = np.array([[data.nitrogeno, data.fosforo, data.potasio, 
                            data.temperatura, data.humedad, data.ph, data.lluvia]])
    
    input_scaled = scaler.transform(input_data)
    prediction = rf_model.predict(input_scaled)
    return {"model": "RandomForest", "Cultivo Recomendado": prediction[0]}
