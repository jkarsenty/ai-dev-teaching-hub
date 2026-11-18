# ML & DL API — FastAPI avancé

API REST avec deux endpoints : sentiment analysis sur texte (TF-IDF) et classification de
chiffres manuscrits (CNN Keras entraîné sur MNIST). Environnement géré avec `uv`.

## Structure du projet

```text
model-deployment/
├── README.md              # ce fichier
├── pyproject.toml         # dépendances (fastapi, scikit-learn, tensorflow, pillow...)
├── uv.lock                # versions figées (à committer)
├── .python-version        # Python utilisé par uv (3.12, comme l'image Docker)
├── Dockerfile
├── .dockerignore
├── app/
│   ├── __init__.py
│   ├── main.py            # routes FastAPI, chargement des modèles
│   ├── schemas.py         # schémas Pydantic (requêtes / réponses)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── text_model.py  # TF-IDF + LogisticRegression
│   │   └── image_model.py # CNN Keras MNIST
│   └── preprocess/
│       ├── __init__.py
│       ├── text.py        # nettoyage du texte
│       └── image.py       # image → tableau 28x28
├── examples/examples/     # images de test (2.png, 4.png, 5.png, mnist_examples.png)
├── .venv/                 # créé par uv (non versionné)
├── text_model.pkl         # généré au 1er lancement (non versionné)
└── image_model.keras      # généré au 1er lancement (non versionné)
```

| Élément | Rôle |
|---|---|
| `app/main.py` | Routes `/health`, `/predict/text`, `/predict/image` |
| `app/schemas.py` | `TextRequest`, `TextResponse`, `ImageResponse` |
| `app/models/` | Création, chargement et prédiction de chaque modèle |
| `app/preprocess/` | Prétraitements séparés des modèles |
| `examples/` | Images pour tester `/predict/image` |

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
   ls pyproject.toml uv.lock app/main.py     # les 3 fichiers doivent s'afficher
   ```
   > ⚠️ `uv` utilise le `pyproject.toml` **le plus proche** en remontant. Depuis la racine du
   > repo, il prendrait le projet racine (sans FastAPI ni TensorFlow). Les modèles
   > (`*.pkl`, `*.keras`) sont aussi créés dans le **dossier courant** : lance toujours depuis
   > `model-deployment/`, sinon ils sont recréés (et ré-entraînés) ailleurs.
4. **Connexion internet au premier lancement** (téléchargement de MNIST, ~11 Mo).
5. **Docker** (partie conteneur uniquement) : Docker Desktop démarré, `docker info` sans erreur.

## Lancement local avec uv

```bash
uv sync                          # crée .venv et installe les dépendances (long : TensorFlow)
uv run uvicorn app.main:app --reload
```

- `uv sync` **crée lui-même** le `.venv` : pas besoin de `uv venv` ni de `source activate`.
- Python 3.12 est choisi via `.python-version` (uv le télécharge s'il est absent).
- Au premier lancement, le CNN est entraîné (~30 s) puis sauvegardé dans
  `image_model.keras` ; les lancements suivants le rechargent directement.
- Avec `--reload`, les modèles sont rechargés à chaque modification de code.

Vérifier le venv :

```bash
uv run python -c "import sys; print(sys.executable)"
# → .../MLOps/model-deployment/.venv/bin/python
```

L'API écoute sur `http://127.0.0.1:8000` (Swagger : `/docs`).

## Lancement avec Docker

Depuis `model-deployment/` :

```bash
docker build -t ml-dl-api .
docker run --rm -p 8000:8000 ml-dl-api
```

Les modèles ne sont pas dans l'image : ils sont créés au démarrage du conteneur
(~30 s avant que l'API réponde) et perdus quand le conteneur est supprimé.

## Tester (local et Docker)

### Swagger

<http://127.0.0.1:8000/docs> → choisir un endpoint → **Try it out** → **Execute**.
Pour `/predict/image`, utiliser le bouton de sélection de fichier.

### curl

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/predict/text \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product"}'

curl -X POST http://127.0.0.1:8000/predict/image \
  -F "file=@examples/examples/2.png"
```

Réponses :

```json
{"text": "I love this product", "sentiment": "positive", "confidence": 0.87}
{"digit": 2, "confidence": 0.98, "probas": {"0": 0.0, "1": 0.0, "2": 0.98, "...": 0.0}}
```

### Bruno / Postman

- `POST http://127.0.0.1:8000/predict/text` : corps **JSON** `{"text": "..."}`.
- `POST http://127.0.0.1:8000/predict/image` : corps **Multipart Form**, champ `file` de type fichier.

### Codes d'erreur

| Cas | Code |
|---|---|
| `text` absent / mauvais type | `422` |
| `text` vide | `400` |
| Fichier ni PNG ni JPEG | `400` |
| Champ `file` absent | `422` |

## Arrêter l'API

`Ctrl+C` dans le terminal. Avec `--reload`, uvicorn démarre un processus fils : si le port
reste occupé, relance la commande `lsof` ci-dessus puis `kill <PID>`.

## Endpoints

| Méthode | Route | Entrée | Réponse |
|---|---|---|---|
| `GET` | `/health` | — | `{"status": "ok"}` |
| `POST` | `/predict/text` | JSON `{"text": "..."}` | `{"text", "sentiment", "confidence"}` |
| `POST` | `/predict/image` | fichier PNG/JPEG (`file`) | `{"digit", "confidence", "probas"}` |
| `GET` | `/docs` | — | Swagger UI |

## Dépannage

| Symptôme | Cause probable |
|---|---|
| `ModuleNotFoundError: app` / `fastapi` | Commande lancée hors de `model-deployment/` |
| Premier démarrage très long | Entraînement du CNN + téléchargement MNIST : normal |
| Les modèles se ré-entraînent à chaque fois | Lancement depuis un autre dossier |
| `Address already in use` (8000) | Relancer la commande `lsof` ; arrêter l'ancien serveur |
| `Cannot connect to the Docker daemon` | Docker Desktop n'est pas démarré |
