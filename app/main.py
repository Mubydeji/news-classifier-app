"""
FastAPI application serving the DistilBERT news classifier.
Model weights are downloaded from Hugging Face Hub at startup, same pattern
as the pneumonia classifier app, keeping the Docker image and repo small.
"""

import os
import logging

import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from huggingface_hub import hf_hub_download

from app.model import load_model_and_tokenizer, predict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_REPO_ID = os.environ.get("MODEL_REPO_ID", "muby26/bbc-news-distilbert")
MODEL_FILENAME = os.environ.get("MODEL_FILENAME", "distilbert_news_classifier.pt")
MIN_TEXT_LENGTH = 20
MAX_TEXT_LENGTH = 20000  # characters; generous but bounded, avoids abuse via huge payloads

app = FastAPI(title="BBC News Category Classifier")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
tokenizer = None


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=MIN_TEXT_LENGTH, max_length=MAX_TEXT_LENGTH)


@app.on_event("startup")
def startup_load_model():
    global model, tokenizer
    logger.info(f"Downloading model weights from {MODEL_REPO_ID}/{MODEL_FILENAME}")
    try:
        weights_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
        model, tokenizer = load_model_and_tokenizer(weights_path, device)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Model failed to load: {e}")
        raise


@app.post("/classify")
async def classify_endpoint(request: ClassifyRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Try again shortly.")

    try:
        predicted_category, probabilities = predict(model, tokenizer, request.text, device)
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail="Classification failed unexpectedly.")

    return {
        "predicted_category": predicted_category,
        "probabilities": probabilities
    }


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}


app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
