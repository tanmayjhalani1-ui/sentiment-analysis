from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference import load_model, predict_sentiment

app = FastAPI(title="Sentiment Analysis API", version="1.0.0")


class SentimentRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)


@app.get("/health")
def health():
    return {"status": "ok", "service": "sentiment-analysis"}


@app.post("/predict")
def predict(request: SentimentRequest):
    try:
        return predict_sentiment(request.text, load_model())
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error))
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))
