from fastapi import APIRouter, HTTPException
import pickle
import numpy as np
from schemas import SoilData

router = APIRouter()

# Cargar los modelos con pickle
try:
    with open("models/svm_model.pkl", "rb") as f:
        svm_model = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Error al cargar los modelos: {e}")

@router.post("/predict/svm")
def predict_svm(data: SoilData):
    try:
        input_data = np.array([[data.nitrogeno, data.fosforo, data.potasio, 
                                data.temperatura, data.humedad, data.ph, data.lluvia]])

        # Verificar que el input tenga 7 características antes de escalar
        if input_data.shape[1] != 7:
            raise ValueError("El input debe contener exactamente 7 características.")

        input_scaled = scaler.transform(input_data)
        prediction = svm_model.predict(input_scaled)

        return {"model": "SVM", "Cultivo Recomendado": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en la predicción: {e}")

