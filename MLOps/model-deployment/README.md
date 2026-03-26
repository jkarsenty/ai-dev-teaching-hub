# ML & DL App — Template Full

Stack complète : API FastAPI (DistilBERT + CNN MNIST) + interface Streamlit.
Les deux services communiquent via HTTP.

## Prérequis
```bash
pip install uv
```

## Lancement local

### 1. Installer les dépendances
```bash
uv venv .venv --python 3.12
source .venv/bin/activate  # Mac / Linux
uv sync
```

### 2. Lancer l'API

Dans un premier terminal depuis `model-deployment/` :
```bash
uv run uvicorn api.main:app --reload
```

> Au premier lancement, DistilBERT est téléchargé (~260MB)
> et le CNN est entraîné sur MNIST (~2 min).

### 3. Lancer le front

Dans un second terminal depuis `model-deployment/` :
```bash
uv run python -m streamlit run front/app.py
```

L'interface est accessible sur `http://localhost:8501`

## Lancement avec Docker Compose
```bash
docker compose up --build
```

- API accessible sur `http://localhost:8000`
- Front accessible sur `http://localhost:8501`

## Endpoints API

### `POST /predict/text`
```bash
curl -X POST http://127.0.0.1:8000/predict/text \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product"}'
```

### `POST /predict/image`
```bash
curl -X POST http://127.0.0.1:8000/predict/image \
  -F "file=@/chemin/vers/image.png"
```

### `GET /docs`
Swagger UI — `http://127.0.0.1:8000/docs`