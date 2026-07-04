import os
import json
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from xgboost import XGBClassifier
from backfill import SYMBOLS

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "xgb_model.json")
FEATURE_ORDER_PATH = os.path.join(MODELS_DIR, "feature_order.json")

ACTION_LABELS = {0: "hold", 1: "buy", 2: "sell"}

_model: XGBClassifier | None = None
_feature_order: list[str] | None = None


def load_model():
    global _model, _feature_order

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Модель не знайдена за шляхом {MODEL_PATH}. Спочатку запусти 3_train.py."
        )

    model = XGBClassifier()
    model.load_model(MODEL_PATH)

    with open(FEATURE_ORDER_PATH) as f:
        feature_order = json.load(f)

    _model = model
    _feature_order = feature_order
    print(f"Модель завантажена. Очікується {len(feature_order)} фічей.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="Trading Decision Inference Service", lifespan=lifespan)


class PredictRequest(BaseModel):
    symbol: str
    features: Dict[str, float]


class PredictResponse(BaseModel):
    symbol: str
    action: str
    confidence: float
    probabilities: Dict[str, float]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}

@app.get("/availableSymbols")
def get_available_symbols():
    """
    Returns the list of symbols that the backfiller is configured to download.
    """
    print(SYMBOLS)
    return {"symbols": SYMBOLS}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if _model is None or _feature_order is None:
        raise HTTPException(status_code=503, detail="Модель ще не завантажена")

    missing = [f for f in _feature_order if f not in request.features]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Відсутні обов'язкові фічі: {missing}. "
                    f"Перевір, чи NestJS рахує ВСІ ті ж індикатори, що й тренування.",
        )

    feature_vector = [[request.features[name] for name in _feature_order]]

    probabilities = _model.predict_proba(feature_vector)[0]
    predicted_class = int(probabilities.argmax())

    return PredictResponse(
        symbol=request.symbol,
        action=ACTION_LABELS[predicted_class],
        confidence=float(probabilities[predicted_class]),
        probabilities={ACTION_LABELS[i]: float(p) for i, p in enumerate(probabilities)},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
