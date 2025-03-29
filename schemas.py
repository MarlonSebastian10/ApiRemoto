from pydantic import BaseModel

class SoilData(BaseModel):
    nitrogeno: float
    fosforo: float
    potasio: float
    temperatura: float
    humedad: float
    ph: float
    lluvia: float
