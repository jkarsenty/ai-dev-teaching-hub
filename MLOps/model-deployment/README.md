# ML & DL API — Template API (Render)

API REST de classification de chiffres manuscrits (CNN Keras, MNIST), prête pour un déploiement
Render. L'endpoint texte (DistilBERT) est désactivé sur cette branche. Environnement géré avec `uv`.

## Structure du projet

```text
model-deployment/
├── README.md               # ce fichier
├── DEPLOIEMENT_RENDER.md   # déploiement sur Render (choix, limites, étapes)
├── render.yaml             # Blueprint Render (Docker, healthcheck /health)
├── pyproject.toml          # fastapi, tensorflow, pillow... (torch/transformers désactivés)
├── uv.lock                 # versions figées (à committer)
├── .python-version         # Python utilisé par uv (3.12, comme l'image Docker)
├── Dockerfile              # entraîne le CNN au build, écoute sur $PORT
├── .dockerignore
├── app/
│   ├── __init__.py
│   ├── main.py            # routes FastAPI, chargement des modèles (lifespan)
│   ├── schemas.py         # schémas Pydantic
│   ├── models/
│   │   ├── __init__.py
│   │   ├── text_model.py  # DistilBERT (sentiment)
│   │   └── image_model.py # CNN Keras MNIST
│   └── preprocess/
│       ├── __init__.py
│       ├── text.py
│       └── image.py
├── models/                 # poids sauvegardés (contenu non versionné, .gitkeep)
├── examples/examples/     # images de test (2.png, 4.png, 5.png, mnist_examples.png)
├── .venv/                  # créé par uv (non versionné)
```

| Élément | Rôle |
|---|---|
| `app/main.py` | Routes `/health` et `/predict/image` (la route texte est commentée) |
| `app/models/text_model.py` | Conservé mais **inutilisé** sur cette branche |
| `Dockerfile` | Entraîne le CNN pendant le build ; `uvicorn` sur `${PORT:-8000}` |
| `render.yaml` | Service `ml-dl-api`, plan `free`, `healthCheckPath: /health` |
| `DEPLOIEMENT_RENDER.md` | Pourquoi / comment déployer, et alternatives |

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
   ls pyproject.toml uv.lock app/main.py render.yaml     # ces fichiers doivent s'afficher
   ```
   > ⚠️ `uv` utilise le `pyproject.toml` **le plus proche** en remontant depuis le dossier
   > courant. Depuis la racine du repo, il prendrait le projet racine (pandas, matplotlib…)
   > et créerait un `.venv` à la racine. Le CNN est sauvegardé dans `models/` **du dossier courant** : lance toujours depuis `model-deployment/`.
4. **Connexion internet au premier lancement** (téléchargement de MNIST, ~11 Mo).
5. **Docker** (build / test conteneur) : Docker Desktop démarré, `docker info` sans erreur.
6. **Render** (déploiement) : voir [`DEPLOIEMENT_RENDER.md`](DEPLOIEMENT_RENDER.md).

## Lancement local avec uv

```bash
uv sync          # crée .venv et installe les dépendances
uv run uvicorn app.main:app --reload
```

- `uv sync` **crée lui-même** le `.venv` : pas besoin de `uv venv` ni de `source activate`.
- Python 3.12 est choisi via `.python-version` (uv le télécharge s'il est absent).
- Au premier lancement, le CNN est entraîné puis sauvegardé dans `models/image_model.keras`.

Vérifier le venv :

```bash
uv run python -c "import sys; print(sys.executable)"
# → .../MLOps/model-deployment/.venv/bin/python
```

L'application écoute sur `http://127.0.0.1:8000`.

## Lancement avec Docker

Depuis `model-deployment/` :

```bash
docker build -t ml-dl-api .
docker run --rm -p 8000:8000 -e PORT=8000 ml-dl-api
```

Contrairement à `template-api`, le CNN est déjà **dans l'image** (entraîné au build) : le conteneur
démarre sans ré-entraînement. Le build est long (installation de TensorFlow + entraînement).

## Tester (local et Docker)

### Swagger

<http://127.0.0.1:8000/docs> → `POST /predict/image` → **Try it out** → **Execute**.

### curl

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/predict/image \
  -F "file=@examples/examples/2.png"
```

Réponse :

```json
{"digit": 2, "confidence": 0.99}
```

### Bruno / Postman

`POST http://127.0.0.1:8000/predict/image` : corps **Multipart Form**, champ `file` de type fichier.

### Codes d'erreur

| Cas | Code |
|---|---|
| Fichier ni PNG ni JPEG | `400` |
| Champ `file` absent | `422` |
| `POST /predict/text` | `404` (endpoint désactivé) |

## Arrêter l'API

`Ctrl+C` dans le terminal. Avec `--reload`, uvicorn démarre un processus fils : si le port
reste occupé, relance la commande `lsof` ci-dessus puis `kill <PID>`.

## Endpoints

| Méthode | Route | Entrée | Réponse |
|---|---|---|---|
| `GET` | `/health` | — | `{"status": "ok"}` |
| `POST` | `/predict/image` | fichier PNG/JPEG (`file`) | `{"digit", "confidence"}` |
| `GET` | `/docs` | — | Swagger UI |

## Dépannage

| Symptôme | Cause probable |
|---|---|
| `ModuleNotFoundError: app` / `fastapi` | Commande lancée hors de `model-deployment/` |
| `Out of memory (used over 512Mi)` sur Render | Plan Free trop petit : voir `DEPLOIEMENT_RENDER.md` |
| `/predict/text` renvoie 404 | Normal : endpoint désactivé sur cette branche |
| `Address already in use` (8000) | Relancer la commande `lsof` ; arrêter l'ancien serveur |
| `Cannot connect to the Docker daemon` | Docker Desktop n'est pas démarré |
