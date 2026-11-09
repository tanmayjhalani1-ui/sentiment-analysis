from __future__ import annotations

import json
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "sentiment_pipeline.joblib"
METADATA_PATH = ROOT / "artifacts" / "metadata.json"
LABELS = ["negative", "neutral", "positive"]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model not found. Run python -m src.train first.")
    return joblib.load(MODEL_PATH)


def predict_sentiment(text: str, model=None) -> dict:
    if not text or not text.strip():
        raise ValueError("Text must not be empty.")
    model = model or load_model()
    prediction = str(model.predict([text])[0])
    result = {"sentiment": prediction, "probabilities": {}}
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([text])[0]
        result["probabilities"] = {
            str(label): float(score) for label, score in zip(model.classes_, probabilities)
        }
        result["confidence"] = float(max(probabilities))
    return result


def load_metadata() -> dict:
    return json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else {}
