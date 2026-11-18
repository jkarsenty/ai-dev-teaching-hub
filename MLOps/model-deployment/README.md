# ML & DL App — Template Full

Stack complète : API FastAPI (DistilBERT + CNN MNIST) + interface Streamlit.
Les deux services communiquent via HTTP. Environnement géré avec `uv`.

## Structure du projet

```text
model-deployment/
├── README.md              # ce fichier
├── pyproject.toml         # dépendances de l'API et du front
├── uv.lock                # versions figées (à committer)
├── .python-version        # Python utilisé par uv (3.12, comme l'image Docker)
├── Dockerfile             # image de l'API
├── docker-compose.yml     # services api + front
├── api/
│   ├── __init__.py
│   ├── main.py            # routes FastAPI (lifespan)
│   ├── schemas.py
│   ├── models/            # text_model.py (DistilBERT), image_model.py (CNN)
│   └── preprocess/        # text.py, image.py
├── front/
│   └── app.py             # interface Streamlit (appelle l'API)
├── models/                # poids sauvegardés (contenu non versionné, .gitkeep)
├── examples/examples/     # images de test (2.png, 4.png, 5.png, mnist_examples.png)
└── .venv/                 # créé par uv (non versionné)
```

| Élément | Rôle |
|---|---|
| `api/` | API FastAPI : `/health`, `/predict/text`, `/predict/image` |
| `front/app.py` | Onglets « Sentiment Analysis » et « MNIST Classifier », requêtes vers l'API |
| `docker-compose.yml` | Services `api` (8000) et `front` (8501) |
| `models/` | `text_model/` et `image_model.keras`, générés au 1er lancement |

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
   ls pyproject.toml uv.lock api/main.py front/app.py     # ces fichiers doivent s'afficher
   ```
   > ⚠️ `uv` utilise le `pyproject.toml` **le plus proche** en remontant depuis le dossier
   > courant. Depuis la racine du repo, il prendrait le projet racine (pandas, matplotlib…)
   > et créerait un `.venv` à la racine. Les modèles sont créés dans `models/` **du dossier courant**.
4. **Connexion internet au premier lancement** (DistilBERT ~260 Mo + MNIST ~11 Mo).
5. **Mémoire** : ≥ 4 Go de RAM libre.
6. **Docker** (partie compose uniquement) : Docker Desktop démarré, `docker info` sans erreur.

## Lancement local avec uv (2 terminaux)

Dans les **deux** terminaux, se placer d'abord dans `MLOps/model-deployment/`.

```bash
uv sync                                 # une seule fois (long : torch + tensorflow)
```

**Terminal 1 — API** (port 8000) :

```bash
uv run uvicorn api.main:app --reload
```

Attendre `Application startup complete` (premier lancement : DistilBERT téléchargé, CNN entraîné).
Vérifier : `curl http://127.0.0.1:8000/health` → `{"status":"ok"}`.

**Terminal 2 — front** (port 8501) :

```bash
uv run python -m streamlit run front/app.py
```

Le front appelle l'API sur `http://localhost:8000` : **lance l'API en premier**.

- Python 3.12 est choisi via `.python-version` (uv le télécharge s'il est absent).
- `uv sync` **crée lui-même** le `.venv` (pas de `uv venv` ni de `source activate`).

## Lancement avec Docker Compose

```bash
docker compose up --build
```

- API : <http://localhost:8000> — front : <http://localhost:8501>

> ⚠️ **À vérifier / corriger** : dans l'état actuel du code, le `Dockerfile` ne copie pas
> `front/` dans l'image et `front/app.py` fixe `API_URL = "http://localhost:8000"` (la variable
> `API_URL=http://api:8000` du compose n'est pas lue). Le service `front` risque donc de ne pas
> fonctionner sous Docker : privilégier le lancement local ci-dessus en attendant.

## Tester

### Front

Ouvrir <http://localhost:8501> : onglet **Sentiment Analysis** (saisir un texte, **Analyser**),
onglet **MNIST Classifier** (dessiner ou uploader `examples/examples/2.png`, **Prédire**).

### API seule (Swagger, curl, Bruno)

<http://127.0.0.1:8000/docs>

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/predict/text \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product"}'

curl -X POST http://127.0.0.1:8000/predict/image \
  -F "file=@examples/examples/2.png"
```

Bruno / Postman : `POST /predict/text` en **JSON** ; `POST /predict/image` en **Multipart Form**
(champ `file`).

### Codes d'erreur

| Cas | Code |
|---|---|
| `text` absent / mauvais type | `422` |
| `text` vide | `400` |
| Fichier ni PNG ni JPEG | `400` |
| Champ `file` absent | `422` |

## Arrêter

`Ctrl+C` dans chaque terminal (front puis API). Avec Docker : `docker compose down`.
Si un port reste occupé (uvicorn `--reload` laisse un processus fils), relancer la commande
`lsof` ci-dessus puis `kill <PID>`.

## Endpoints de l'API

| Méthode | Route | Entrée | Réponse |
|---|---|---|---|
| `GET` | `/health` | — | `{"status": "ok"}` |
| `POST` | `/predict/text` | JSON `{"text": "..."}` | `{"text", "sentiment", "confidence"}` |
| `POST` | `/predict/image` | fichier PNG/JPEG (`file`) | `{"digit", "confidence"}` |
| `GET` | `/docs` | — | Swagger UI |

## Dépannage

| Symptôme | Cause probable |
|---|---|
| Front : « Impossible de contacter l'API » | API non démarrée (terminal 1) ou port 8000 occupé |
| `ModuleNotFoundError: api` | Commande lancée hors de `model-deployment/` |
| Premier démarrage très long | DistilBERT + entraînement du CNN : normal |
| Processus tué / `OOM` | Pas assez de RAM (torch + TensorFlow) |
| `Address already in use` (8000 / 8501) | Relancer la commande `lsof` ; arrêter l'ancien serveur |
