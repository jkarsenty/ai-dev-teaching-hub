# MNIST Classifier — Streamlit Front (Render)

Interface web de classification de chiffres manuscrits. Appelle l'API `ml-dl-api`
(branche `mlops/template-api-render`) via `POST /predict/image`. Environnement géré avec `uv`.

## Structure du projet

```text
model-deployment/
├── README.md               # ce fichier
├── DEPLOIEMENT_RENDER.md   # déploiement sur Render (choix, étapes)
├── render.yaml             # Blueprint Render (variable API_URL)
├── pyproject.toml          # streamlit, streamlit-drawable-canvas, pillow, requests
├── uv.lock                 # versions figées (à committer)
├── .python-version         # Python utilisé par uv (3.12, comme l'image Docker)
├── Dockerfile              # lance Streamlit sur $PORT
├── .dockerignore
├── app.py                  # interface Streamlit qui appelle l'API
├── model.py                # hérité de front-simple, non utilisé ici
├── models/                 # .gitkeep (non utilisé : pas de modèle local)
├── examples/examples/     # images de test (2.png, 4.png, 5.png, mnist_examples.png)
└── .venv/                  # créé par uv (non versionné)
```

| Fichier | Rôle |
|---|---|
| `app.py` | Onglets Dessiner / Uploader ; envoie l'image à `{API_URL}/predict/image` |
| `render.yaml` | Service `mnist-front`, `API_URL=https://ml-dl-api.onrender.com` |
| `Dockerfile` | `streamlit run app.py` sur `${PORT:-8501}` |

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
   ls pyproject.toml uv.lock app.py     # ces fichiers doivent s'afficher
   ```
   > ⚠️ `uv` utilise le `pyproject.toml` **le plus proche** en remontant depuis le dossier
   > courant. Depuis la racine du repo, il prendrait le projet racine (pandas, matplotlib…)
   > et créerait un `.venv` à la racine. 
4. **Une API MNIST joignable** (voir ci-dessous) : sans elle, le front affiche « Erreur en appelant l'API ».
5. **Docker** (partie conteneur uniquement) : Docker Desktop démarré, `docker info` sans erreur.

## Obtenir une API

### Option A — API locale (branche `mlops/template-api-render`)

Le front et l'API sont sur **deux branches différentes** : utilise un second dossier (worktree).
Depuis la racine du repo :

```bash
git worktree add ../mlops-api-render mlops/template-api-render
cd ../mlops-api-render/MLOps/model-deployment
uv sync
uv run uvicorn app.main:app          # port 8000
```

Vérifier : `curl http://127.0.0.1:8000/health` → `{"status":"ok"}`.

### Option B — API déjà déployée sur Render

Utiliser son URL (le premier appel peut être lent : démarrage à froid du plan Free) :

```bash
export API_URL=https://ml-dl-api.onrender.com
```

## Lancement local avec uv

Dans **un autre terminal**, depuis `MLOps/model-deployment/` de cette branche :

```bash
uv sync
API_URL=http://localhost:8000 uv run python -m streamlit run app.py
```

- Sans `API_URL`, l'application utilise `http://localhost:8000` par défaut.
- Python 3.12 est choisi via `.python-version` (uv le télécharge s'il est absent).

L'interface est accessible sur <http://localhost:8501> ; l'URL de l'API utilisée est affichée
sous le titre (« API : … »).

## Lancement avec Docker

```bash
docker build -t mnist-front .
docker run --rm -p 8501:8501 -e PORT=8501 \
  -e API_URL=http://host.docker.internal:8000 mnist-front
```

`host.docker.internal` permet au conteneur d'appeler une API lancée sur ton Mac (option A).
Pour l'API Render, mets son URL dans `API_URL`.

## Tester

1. Ouvrir <http://localhost:8501>.
2. Onglet **Dessiner** : tracer un chiffre, puis **Prédire** ; ou onglet **Uploader** avec
   `examples/examples/2.png`.
3. Le chiffre prédit et la confiance s'affichent. API arrêtée : message d'erreur rouge, sans plantage.

```bash
curl http://localhost:8501/_stcore/health     # → ok
```

## Arrêter l'application

`Ctrl+C` dans le terminal. Si le port 8501 reste occupé, relance la commande `lsof`
ci-dessus puis `kill <PID>`.

Penser aussi à arrêter l'API (option A) avec `Ctrl+C`.

## Dépannage

| Symptôme | Cause probable |
|---|---|
| « Erreur en appelant l'API » | API non démarrée, mauvaise `API_URL`, ou API Render en démarrage à froid |
| `Connection refused` depuis Docker | Utiliser `host.docker.internal` au lieu de `localhost` |
| `ModuleNotFoundError: streamlit` | Commande lancée hors de `model-deployment/` |
| `Port 8501 is already in use` | Relancer la commande `lsof` ; arrêter l'ancien Streamlit |
| Le canvas de dessin ne s'affiche pas | Incompatibilité possible `streamlit-drawable-canvas` / Streamlit : utiliser l'onglet **Uploader** |
