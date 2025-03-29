from fastapi import FastAPI
from routes import neural_network, svm, random_forest

app = FastAPI(title="API de Recomendación de Cultivos")

app.include_router(neural_network.router)
app.include_router(svm.router)
app.include_router(random_forest.router)

@app.get("/")
def read_root():
    return {"message": "API para recomendación de cultivos con IA"}
