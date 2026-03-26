from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import load_model, predict

app = FastAPI(
    title="Sentiment Analysis API",
    description="Analyse de sentiment basée sur un pipeline TF-IDF + Régression Logistique",
    version="1.0.0",
)

model = load_model()

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict_route(request: PredictRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Le champ 'text' ne peut pas être vide")
    result = predict(model, request.text)
    return PredictResponse(text=request.text, **result)