from fastapi import FastAPI, HTTPException, UploadFile, File
from contextlib import asynccontextmanager

from app.schemas import TextRequest, TextResponse, ImageResponse
from app.models.text_model import load_text_model, predict_text
from app.models.image_model import load_image_model, predict_image

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    models["text"] = load_text_model()
    models["image"] = load_image_model()
    yield
    models.clear()

app = FastAPI(
    title="ML & DL API",
    description="Sentiment analysis via DistilBERT et classification MNIST via CNN Keras",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict/text", response_model=TextResponse)
def predict_text_route(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Le champ 'text' ne peut pas être vide")
    result = predict_text(models["text"], request.text)
    return TextResponse(text=request.text, **result)

@app.post("/predict/image", response_model=ImageResponse)
async def predict_image_route(file: UploadFile = File(...)):
    if file.content_type not in ["image/png", "image/jpeg"]:
        raise HTTPException(status_code=400, detail="Format accepté : PNG ou JPEG")
    image_bytes = await file.read()
    result = predict_image(models["image"], image_bytes)
    return ImageResponse(**result)