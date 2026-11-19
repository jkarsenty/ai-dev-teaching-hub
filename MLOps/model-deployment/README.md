# Sentiment Analysis API — Flask + uv + Docker

API REST de sentiment analysis (TF-IDF + régression logistique) servie avec Flask,
environnement géré avec `uv`, conteneurisée avec Docker.

## Structure du projet

```text
model-deployment/
├── README.md            # ce fichier
├── pyproject.toml       # dépendances (flask, scikit-learn) + groupe dev
├── uv.lock              # versions figées (à committer)
├── .python-version      # Python utilisé par uv (3.11, comme l'image Docker)
├── Dockerfile           # image de l'API
├── .dockerignore        # exclut .venv, *.pkl... du contexte de build
├── app.py               # API Flask (/predict, /health)
├── model.py             # preprocess, entraînement, chargement, prédiction
├── .venv/               # créé par uv (non versionné)
└── model.pkl            # généré au 1er lancement (non versionné)
```

| Fichier | Rôle |
|---|---|
| `pyproject.toml` / `uv.lock` | Définition et verrouillage des dépendances |
| `Dockerfile` | `python:3.11-slim` + `uv sync --frozen --no-dev` ; lance `python app.py` |
| `app.py` / `model.py` | Identiques à `flask-avance` |

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
   > ⚠️ Pourquoi : `uv` utilise le `pyproject.toml` **le plus proche** en remontant
   > depuis le dossier courant. Depuis la racine du repo, il prendrait le `pyproject.toml`
   > du projet racine (pandas, matplotlib…, sans Flask) et créerait un `.venv` à la racine.
   > Et `model.pkl` est créé dans le dossier courant.
   > possible que le uv.lock apparaisse pas si c'est la première fois que tu lances uv, il sera créé après le premier `uv sync`.
   
4. **Docker** (uniquement pour la partie conteneur) : Docker Desktop démarré, `docker info` sans erreur.

## Lancement local avec uv

```bash
uv sync               # crée model-deployment/.venv et installe les dépendances
uv run python app.py  # lance l'API
```

- `uv sync` **crée lui-même** le `.venv` : pas besoin de `uv venv` ni de `source activate`.
- `uv run` utilise automatiquement ce `.venv`.
- Python 3.11 est choisi via `.python-version` (uv le télécharge s'il est absent).

Vérifier où est le venv et quel Python est utilisé :

```bash
uv run python -c "import sys; print(sys.executable)"
# → .../MLOps/model-deployment/.venv/bin/python
```

Variante : activer le venv à la main.

```bash
source .venv/bin/activate    # Windows : .venv\Scripts\activate
python app.py
deactivate
```

L'API écoute sur `http://127.0.0.1:5000`.

## Lancement avec Docker

Depuis `model-deployment/` :

```bash
docker build -t sentiment-api .
docker run --rm -p 5000:5000 sentiment-api
```

L'image est construite avec les versions de `uv.lock` (`--frozen`). `model.pkl` est
créé dans le conteneur au premier démarrage (il n'est pas dans l'image).

## Tester (local et Docker)

```bash
curl http://127.0.0.1:5000/health

curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product"}'
```

Réponse :

```json
{"text": "I love this product", "sentiment": "positive", "confidence": 0.87}
```

Sans champ `text`, l'API répond `400`.

## Arrêter l'API

`Ctrl+C` dans le terminal. En mode `debug=True`, Flask démarre un processus fils (reloader) :
si le port reste occupé après l'arrêt, relance la commande `lsof` ci-dessus puis `kill <PID>`.

## Endpoints

| Méthode | Route | Corps | Réponse |
|---|---|---|---|
| `GET` | `/health` | — | `{"status": "ok"}` |
| `POST` | `/predict` | `{"text": "..."}` | `{"text", "sentiment", "confidence"}` |

## Dépannage

| Symptôme | Cause probable |
|---|---|
| `ModuleNotFoundError: flask` | Commande lancée hors de `model-deployment/` (mauvais `pyproject.toml`) |
| `Address already in use` / port 5000 | Relancer la commande `lsof` ; AirPlay Receiver sur macOS |
| `Cannot connect to the Docker daemon` | Docker Desktop n'est pas démarré |
