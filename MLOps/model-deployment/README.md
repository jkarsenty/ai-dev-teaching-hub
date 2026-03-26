# ML & DL API — Template API

API REST complète avec deux endpoints : sentiment analysis via DistilBERT et
classification de chiffres manuscrits via CNN Keras (MNIST).

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

> Au premier lancement, le modèle DistilBERT est téléchargé (~260MB).  
> et le CNN est entraîné sur MNIST (~2 min). Les modèles sont ensuite.  
> sauvegardés dans `models/` et rechargés directement aux lancements suivants.  

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
{"text": "I love this product", "sentiment": "positive", "confidence": 0.9998}
```

**Test curl :**
```bash
curl -X POST http://127.0.0.1:8000/predict/text \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product"}'
```

### `POST /predict/image`
Upload d'une image PNG ou JPEG d'un chiffre manuscrit.

**Réponse :**
```json
{"digit": 7, "confidence": 0.9981}
```

**Test curl :**
```bash
curl -X POST http://127.0.0.1:8000/predict/image \
  -F "file=@/chemin/vers/image.png"
```

### `GET /docs`
Swagger UI — `http://127.0.0.1:8000/docs`