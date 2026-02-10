from pydantic import BaseModel
from typing import Dict

class ImagePrediction(BaseModel):
    Number: int
    Proba: dict

class SentimentPrediction(BaseModel):
    Proba: Dict[str, float]
    Sentiment: str
    Confidence: float

class SentimentRequest(BaseModel):
    text: str
