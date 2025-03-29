from fastapi import APIRouter
import joblib
import numpy as np
from schemas import SoilData

router = APIRouter()


mlp_model = joblib.load("models/mlp_model.pkl")
scaler = joblib.load("models/scaler.pkl")

@router.post("/predict/neural_network")
def predict_nn(data: SoilData):
    input_data = np.array([[data.nitrogeno, data.fosforo, data.potasio, 
                            data.temperatura, data.humedad, data.ph, data.lluvia]])
    
    input_scaled = scaler.transform(input_data)
    prediction = mlp_model.predict(input_scaled)
    return {"model": "Neural Network", "Cultivo Recomendado": prediction[0]}
