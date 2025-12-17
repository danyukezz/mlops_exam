from fastapi import FastAPI
import pandas as pd
import mlflow.pyfunc

from app.schemas import PredictRequest, PredictResponse

MODEL_DIR = "model"   # folder containing MLmodel
model = mlflow.pyfunc.load_model(MODEL_DIR)

app = FastAPI(title="GOT House Predictor", version="1.0")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    X = pd.DataFrame([payload.model_dump()])
    y = model.predict(X)

    # MLflow pyfunc often returns numpy array / pandas series / dataframe depending on flavor
    if hasattr(y, "iloc"):
        pred = y.iloc[0]
    else:
        pred = y[0]

    return {"house": str(pred)}
