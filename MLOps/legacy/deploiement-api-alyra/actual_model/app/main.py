# import des librairies
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from schemas import ImagePrediction, SentimentRequest, SentimentPrediction

from image_preprocess import preprocess_image
from image_model import load_model, predict

from bert_model import Model_BERT, get_bert_model

# creation de l'instance fastAPI
app = FastAPI()

# chargement du modèle au démarrage de l'Application
model = load_model("model/model_alyra_0.1.0.h5")

# création des routes 
@app.get("/")
def root():
    return {"message": "Hello World"}

@app.get('/health')
def health_check():
    return {"status": "ok", "message": "API is running smoothly.", "config": "uv+pyproject.toml"}

@app.post("/predict_image", response_model=ImagePrediction)
async def predict_image(file: UploadFile = File(...)):
    try :
        image = await file.read()
        preprocessed_image = preprocess_image(image)
        prediction = predict(model, preprocessed_image)
        round_prediction = {i:round(p, 2) for i,p in enumerate(prediction[0])}
        
        return {
            "Number" : np.argmax(prediction[0]),
            "Proba": round_prediction
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail="Error reading image file")

@app.post("/predict_sentiment", response_model=SentimentPrediction)
async def predict_bert(request:SentimentRequest, model:Model_BERT = Depends(get_bert_model)):
    
    sentiment,confidence, proba = model.predict(request.text)

    return {
        "Proba": proba,
        "Sentiment": sentiment,
        "Confidence": confidence
    }