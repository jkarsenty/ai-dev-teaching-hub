# ML & DL API — FastAPI Avancé

API REST avec deux endpoints : sentiment analysis sur texte et classification de chiffres manuscrits (MNIST).

## Prérequis
```bash
pip install uv
```

## Lancement local avec uv
```bash
uv venv .venv --python 3.12
source .venv/bin/activate  # Mac / Linux
uv sync
uv run uvicorn app.main:app --reload
```

## Lancement avec Docker
```bash
docker build -t ml-dl-api .
docker run -p 8000:8000 ml-dl-api
```

## Endpoints

### `GET /health`
```json
{"status": "ok"}
```

### `POST /predict/text`

**Body JSON :**
```json
{"text": "I love this product"}
```

**Réponse :**
```json
{"text": "I love this product", "sentiment": "positive", "confidence": 0.87}
```

**Test curl :**
```bash
curl -X POST http://127.0.0.1:8000/predict/text \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product"}'
```

### `POST /predict/image`
Upload d'une image PNG ou JPEG d'un chiffre manuscrit (28x28 recommandé).

**Réponse :**
```json
{"digit": 7, "confidence": 0.98}
```

**Test curl :**
```bash
curl -X POST http://127.0.0.1:8000/predict/image \
  -F "file=@/chemin/vers/image.png"
```

### `GET /docs`
Swagger UI — `http://127.0.0.1:8000/docs`