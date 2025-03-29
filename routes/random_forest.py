from fastapi import APIRouter, HTTPException
import pickle
import numpy as np
from schemas import SoilData

router = APIRouter()

# Cargar los modelos con pickle
try:
    with open("models/random_forest_model.pkl", "rb") as f:
        rf_model = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Error al cargar los modelos: {e}")

@router.post("/predict/random_forest")
def predict_rf(data: SoilData):
    try:
        input_data = np.array([[data.nitrogeno, data.fosforo, data.potasio, 
                                data.temperatura, data.humedad, data.ph, data.lluvia]])

        # Asegurar que el input tenga el formato correcto antes de transformarlo
        if input_data.shape[1] != 7:
            raise ValueError("El input debe contener exactamente 7 características.")

        input_scaled = scaler.transform(input_data)
        prediction = rf_model.predict(input_scaled)

        return {"model": "Random Forest", "Cultivo Recomendado": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en la predicción: {e}")

