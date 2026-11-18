# ML & DL API — Template API

API REST avec deux endpoints : sentiment analysis via **DistilBERT** et classification de
chiffres manuscrits via **CNN Keras** (MNIST). Environnement géré avec `uv`.

## Structure du projet

```text
model-deployment/
├── README.md              # ce fichier
├── pyproject.toml         # fastapi, tensorflow, transformers, torch, pillow...
├── uv.lock                # versions figées (à committer)
├── .python-version        # Python utilisé par uv (3.12, comme l'image Docker)
├── Dockerfile
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
├── models/                # poids sauvegardés (contenu non versionné, .gitkeep)
├── examples/examples/     # images de test (2.png, 4.png, 5.png, mnist_examples.png)
├── .venv/                 # créé par uv (non versionné)
```

| Élément | Rôle |
|---|---|
| `app/main.py` | Routes `/health`, `/predict/text`, `/predict/image` ; chargement des modèles au démarrage |
| `app/models/` | Création / chargement / prédiction de chaque modèle |
| `app/preprocess/` | Nettoyage du texte, redimensionnement de l'image |
| `models/` | `text_model/` (DistilBERT) et `image_model.keras` (CNN), générés au 1er lancement |
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
   ls pyproject.toml uv.lock app/main.py     # ces fichiers doivent s'afficher
   ```
   > ⚠️ `uv` utilise le `pyproject.toml` **le plus proche** en remontant depuis le dossier
   > courant. Depuis la racine du repo, il prendrait le projet racine (pandas, matplotlib…)
   > et créerait un `.venv` à la racine. Les modèles sont créés dans `models/` **du dossier courant** : lance toujours depuis `model-deployment/`.
4. **Connexion internet au premier lancement** (DistilBERT ~260 Mo + MNIST ~11 Mo).
5. **Mémoire** : ≥ 4 Go de RAM libre (torch + TensorFlow). Pour Docker, régler la mémoire de Docker Desktop en conséquence.
6. **Docker** (partie conteneur uniquement) : Docker Desktop démarré, `docker info` sans erreur.

## Lancement local avec uv

```bash
uv sync          # crée .venv et installe les dépendances
uv run uvicorn app.main:app --reload
```

- `uv sync` **crée lui-même** le `.venv` : pas besoin de `uv venv` ni de `source activate`.
- Python 3.12 est choisi via `.python-version` (uv le télécharge s'il est absent).
- Au premier lancement : DistilBERT est téléchargé et le CNN entraîné, puis tout est sauvegardé
  dans `models/` ; les lancements suivants rechargent directement (plus rapide).

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
docker run --rm -p 8000:8000 ml-dl-api
```

Les modèles ne sont pas dans l'image : ils sont téléchargés / entraînés au démarrage du
conteneur, et perdus quand il est supprimé.

## Tester (local et Docker)

### Swagger

<http://127.0.0.1:8000/docs> → choisir un endpoint → **Try it out** → **Execute**.

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
{"text": "I love this product", "sentiment": "positive", "confidence": 0.9998}
{"digit": 2, "confidence": 0.99}
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
| `POST` | `/predict/image` | fichier PNG/JPEG (`file`) | `{"digit", "confidence"}` |
| `GET` | `/docs` | — | Swagger UI |

## Dépannage

| Symptôme | Cause probable |
|---|---|
| `ModuleNotFoundError: app` / `fastapi` | Commande lancée hors de `model-deployment/` |
| Premier démarrage très long | Téléchargement de DistilBERT + entraînement du CNN : normal |
| Le modèle texte est re-téléchargé à chaque fois | Lancement depuis un autre dossier (`models/` introuvable) |
| Processus tué / `OOM` | Pas assez de RAM (torch + TensorFlow) |
| `Address already in use` (8000) | Relancer la commande `lsof` ; arrêter l'ancien serveur |
| `Cannot connect to the Docker daemon` | Docker Desktop n'est pas démarré |
