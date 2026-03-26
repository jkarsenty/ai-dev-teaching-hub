import os
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, Pipeline
import torch

from api.preprocess.text import preprocess_text

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
MODEL_PATH = "models/text_model"

LABEL_MAP = {
    "POSITIVE": "positive",
    "NEGATIVE": "negative",
}

def create_text_model() -> Pipeline:
    os.makedirs(MODEL_PATH, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(MODEL_PATH)
    model.save_pretrained(MODEL_PATH)
    print("Modèle texte téléchargé et sauvegardé.")
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def load_text_model() -> Pipeline:
    if not os.path.exists(MODEL_PATH):
        print("Aucun modèle texte trouvé, téléchargement en cours...")
        return create_text_model()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    print("Modèle texte chargé.")
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def predict_text(model: Pipeline, text: str) -> dict:
    cleaned = preprocess_text(text)
    result = model(cleaned)[0]
    label = LABEL_MAP.get(result["label"], result["label"].lower())
    confidence = round(float(result["score"]), 4)
    return {"sentiment": label, "confidence": confidence}