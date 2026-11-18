# Sentiment Analysis API — FastAPI + uv + Docker

API REST de sentiment analysis (TF-IDF + régression logistique) servie avec FastAPI,
environnement géré avec `uv`, conteneurisée avec Docker.

## Structure du projet

```text
model-deployment/
├── README.md            # ce fichier
├── pyproject.toml       # dépendances (fastapi, uvicorn, scikit-learn) + groupe dev
├── uv.lock              # versions figées (à committer)
├── .python-version      # Python utilisé par uv (3.12, comme l'image Docker)
├── Dockerfile           # image de l'API
├── .dockerignore        # exclut .venv, *.pkl... du contexte de build
├── app.py               # API FastAPI (/predict, /health, /docs)
├── model.py             # preprocess, entraînement, chargement, prédiction
├── .venv/               # créé par uv (non versionné)
└── model.pkl            # généré au 1er lancement (non versionné)
```

| Fichier | Rôle |
|---|---|
| `pyproject.toml` / `uv.lock` | Définition et verrouillage des dépendances |
| `Dockerfile` | `python:3.12-slim` + `uv sync --frozen --no-dev` ; lance `uvicorn` sur le port 8000 |
| `app.py` | Routes FastAPI, schémas Pydantic (`PredictRequest`, `PredictResponse`) |
| `model.py` | Identique à `flask-avance-uv-docker` |

## Prérequis

1. **Ports libres** (aucun résultat attendu) :
   ```bash
   lsof -nP -iTCP:5000,5001,8000,8501 -sTCP:LISTEN
   ```
2. **`uv` installé** : `uv --version` (installation : `brew install uv`).
3. **Être dans `MLOps/model-deployment/`**, pas à la racine du repo. Le chemin contient
   des espaces : mets-le entre guillemets.
   ```bash
   cd "<chemin du repo>/MLOps/model-deployment"
   ls pyproject.toml uv.lock app.py     # les 3 fichiers doivent s'afficher
   ```
   > ⚠️ `uv` utilise le `pyproject.toml` **le plus proche** en remontant depuis le dossier
   > courant. Depuis la racine du repo, il prendrait le projet racine (pandas, matplotlib…,
   > sans FastAPI) et créerait un `.venv` à la racine. Et `model.pkl` est créé dans le
   > dossier courant.
4. **Docker** (uniquement pour la partie conteneur) : Docker Desktop démarré, `docker info` sans erreur.

## Lancement local avec uv

```bash
uv sync                           # crée model-deployment/.venv et installe les dépendances
uv run uvicorn app:app --reload   # lance l'API
```

- `uv sync` **crée lui-même** le `.venv` : pas besoin de `uv venv` ni de `source activate`.
- `uv run` utilise automatiquement ce `.venv`.
- Python 3.12 est choisi via `.python-version` (uv le télécharge s'il est absent).

Vérifier où est le venv et quel Python est utilisé :

```bash
uv run python -c "import sys; print(sys.executable)"
# → .../MLOps/model-deployment/.venv/bin/python
```

Variante : activer le venv à la main.

```bash
source .venv/bin/activate    # Windows : .venv\Scripts\activate
uvicorn app:app --reload
deactivate
```

L'API écoute sur `http://127.0.0.1:8000`.

## Lancement avec Docker

Depuis `model-deployment/` :

```bash
docker build -t sentiment-api .
docker run --rm -p 8000:8000 sentiment-api
```

L'image est construite avec les versions de `uv.lock` (`--frozen`). `model.pkl` est
créé dans le conteneur au premier démarrage (il n'est pas dans l'image).

## Tester (local et Docker)

### Swagger

Ouvrir <http://127.0.0.1:8000/docs> → `POST /predict` → **Try it out** → **Execute**.

### curl

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product"}'
```

Réponse :

```json
{"text": "I love this product", "sentiment": "positive", "confidence": 0.87}
```

### Bruno / Postman

`POST http://127.0.0.1:8000/predict`, corps **JSON** : `{"text": "I love this product"}`.

### Codes d'erreur

| Cas | Code |
|---|---|
| Champ `text` absent ou mauvais type | `422` (validation Pydantic) |
| `text` vide ou espaces | `400` |

## Arrêter l'API

`Ctrl+C` dans le terminal. Avec `--reload`, uvicorn démarre un processus fils : si le port
reste occupé après l'arrêt, relance la commande `lsof` ci-dessus puis `kill <PID>`.

## Endpoints

| Méthode | Route | Corps | Réponse |
|---|---|---|---|
| `GET` | `/health` | — | `{"status": "ok"}` |
| `POST` | `/predict` | `{"text": "..."}` | `{"text", "sentiment", "confidence"}` |
| `GET` | `/docs` | — | Swagger UI |

## Dépannage

| Symptôme | Cause probable |
|---|---|
| `ModuleNotFoundError: fastapi` | Commande lancée hors de `model-deployment/` (mauvais `pyproject.toml`) |
| `Address already in use` (8000) | Relancer la commande `lsof` ; arrêter l'ancien serveur |
| `Cannot connect to the Docker daemon` | Docker Desktop n'est pas démarré |
