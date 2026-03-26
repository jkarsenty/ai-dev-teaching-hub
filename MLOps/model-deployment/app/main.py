from fastapi import FastAPI, HTTPException, UploadFile, File
from app.schemas import TextRequest, TextResponse, ImageResponse
from app.models.text_model import load_text_model, predict_text
from app.models.image_model import load_image_model, predict_image

app = FastAPI(
    title="ML & DL API",
    description="Sentiment analysis (texte) et classification de chiffres (image MNIST)",
    version="1.0.0",
)

text_model = load_text_model()
image_model = load_image_model()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict/text", response_model=TextResponse)
def predict_text_route(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Le champ 'text' ne peut pas être vide")
    result = predict_text(text_model, request.text)
    return TextResponse(text=request.text, **result)

@app.post("/predict/image", response_model=ImageResponse)
async def predict_image_route(file: UploadFile = File(...)):
    if file.content_type not in ["image/png", "image/jpeg"]:
        raise HTTPException(status_code=400, detail="Format accepté : PNG ou JPEG")
    image_bytes = await file.read()
    result = predict_image(image_model, image_bytes)
    return ImageResponse(**result)