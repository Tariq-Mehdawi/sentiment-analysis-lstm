"""
FastAPI service for the BiLSTM sentiment model.

Run locally:
    uvicorn src.app.main:app --host 0.0.0.0 --port 8000
"""

import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from src.predict import SentimentPredictor

MODEL_DIR = os.getenv("MODEL_DIR", "models")
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model + tokenizer ONCE at startup, not on every request
    state["predictor"] = SentimentPredictor(model_dir=MODEL_DIR)
    yield
    state.clear()


app = FastAPI(title="Sentiment Analysis API", version="1.0.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    text: str = Field(
        ...,
        min_length=1,
        max_length=20000,
        examples=["This movie completely blew me away."],
    )


class PredictResponse(BaseModel):
    label: Literal["POSITIVE", "NEGATIVE"]
    score: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.5, le=1.0)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "predictor" in state}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    result = state["predictor"].predict(request.text)
    return PredictResponse(
        label=result["label"],
        score=result["score"],
        confidence=result["confidence"],
    )