# Sentiment Analysis API — FastAPI + uv + Docker

API REST de sentiment analysis basée sur un pipeline TF-IDF + Régression Logistique,
servie avec FastAPI, gérée avec uv et conteneurisée avec Docker.

## Prérequis

Installer uv si nécessaire :
```bash
pip install uv
```

## Lancement local avec uv

Créer un environnement virtuel dédié avec Python 3.12 :
```bash
uv venv .venv --python 3.12
```

Activer l'environnement :
```bash
# Mac / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

Installer les dépendances :
```bash
uv sync
```

Lancer l'API :
```bash
uv run uvicorn app:app --reload
```

## Lancement avec Docker

Build :
```bash
docker build -t sentiment-api .
```

Run :
```bash
docker run -p 8000:8000 sentiment-api
```

## Endpoints

### `POST /predict`

**Body JSON :**
```json
{"text": "I love this product"}
```

**Réponse :**
```json
{"text": "I love this product", "sentiment": "positive", "confidence": 0.87}
```

### `GET /health`
```json
{"status": "ok"}
```

### `GET /docs`
Interface Swagger UI auto-générée — accessible sur `http://127.0.0.1:8000/docs`