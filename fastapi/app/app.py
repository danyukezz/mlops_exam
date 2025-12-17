import os
from typing import Any

import pandas as pd
from fastapi import FastAPI

from app.schemas import PredictRequest, PredictResponse

import mlflow.pyfunc


app = FastAPI(title="GOT House Predictor", version="1.0")


def _model_dir() -> str:
    here = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(here, "..", "model"))


MODEL_DIR = _model_dir()

# Load MLflow model (pyfunc wrapper)
model = mlflow.pyfunc.load_model(MODEL_DIR)

# Get underlying sklearn model so we can access feature_names_in_
# (This is needed because you trained on one-hot columns.)
sk_model: Any = getattr(model, "_model_impl", None)
if sk_model is not None:
    sk_model = getattr(sk_model, "sklearn_model", None)

if sk_model is None or not hasattr(sk_model, "feature_names_in_"):
    raise RuntimeError(
        "Could not access sklearn_model.feature_names_in_. "
        "This model must be a sklearn model trained on one-hot encoded columns."
    )

TRAIN_COLS = list(sk_model.feature_names_in_)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_dir": MODEL_DIR,
        "num_train_cols": len(TRAIN_COLS),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    raw = pd.DataFrame([payload.model_dump()])

    # Must match your Azure prep step: one-hot + dummy_na, then align columns
    X = pd.get_dummies(raw, dummy_na=True)
    X = X.reindex(columns=TRAIN_COLS, fill_value=0)

    y = model.predict(X)
    pred = y.iloc[0] if hasattr(y, "iloc") else y[0]

    return PredictResponse(house=str(pred), model_loaded=True)
