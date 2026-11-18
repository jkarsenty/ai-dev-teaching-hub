# Sentiment Analysis — Streamlit simple

Interface web de sentiment analysis avec un modèle TF-IDF + régression logistique embarqué
directement dans le front, sans API. Environnement géré avec `uv`.

## Structure du projet

```text
model-deployment/
├── README.md            # ce fichier
├── pyproject.toml       # dépendances (streamlit, scikit-learn)
├── uv.lock              # versions figées (à committer)
├── .python-version      # Python utilisé par uv (3.12)
├── app.py               # interface Streamlit
├── model.py             # preprocess, entraînement, chargement, prédiction
├── examples/examples/     # images MNIST (non utilisées sur cette branche)
├── .venv/               # créé par uv (non versionné)
└── model.pkl            # généré au 1er lancement (non versionné)
```

| Fichier | Rôle |
|---|---|
| `app.py` | Interface Streamlit : saisie du texte, affichage du sentiment et de la confiance |
| `model.py` | `create_model()`, `load_model()`, `predict()` ; modèle sérialisé avec `pickle` |
| `model.pkl` | Modèle sauvegardé. Supprime-le pour le ré-entraîner |

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
   ls pyproject.toml uv.lock app.py model.py     # ces fichiers doivent s'afficher
   ```
   > ⚠️ `uv` utilise le `pyproject.toml` **le plus proche** en remontant depuis le dossier
   > courant. Depuis la racine du repo, il prendrait le projet racine (pandas, matplotlib…)
   > et créerait un `.venv` à la racine. Et `model.pkl` est créé dans le dossier courant.

## Lancement local avec uv

```bash
uv sync          # crée .venv et installe les dépendances
uv run streamlit run app.py
```

- `uv sync` **crée lui-même** le `.venv` : pas besoin de `uv venv` ni de `source activate`.
- Python 3.12 est choisi via `.python-version` (uv le télécharge s'il est absent).
- Au premier lancement, Streamlit peut demander un e-mail : appuie sur Entrée pour passer
  (ou lance avec `--server.headless=true`, qui n'ouvre pas non plus le navigateur).
- Le navigateur s'ouvre automatiquement.

Vérifier le venv :

```bash
uv run python -c "import sys; print(sys.executable)"
# → .../MLOps/model-deployment/.venv/bin/python
```

L'application écoute sur `http://localhost:8501`.

## Tester

1. Ouvrir <http://localhost:8501>.
2. Saisir un texte, par exemple `I love this product`, puis cliquer sur **Analyser**.
3. Le sentiment (😊 positive / 😞 negative) et la confiance s'affichent.

Test rapide en ligne de commande (santé du serveur) :

```bash
curl http://localhost:8501/_stcore/health     # → ok
```

Un texte vide affiche un avertissement : « Veuillez entrer un texte. »

## Arrêter l'application

`Ctrl+C` dans le terminal. Si le port 8501 reste occupé, relance la commande `lsof`
ci-dessus puis `kill <PID>`.

## Dépannage

| Symptôme | Cause probable |
|---|---|
| `ModuleNotFoundError: streamlit` | Commande lancée hors de `model-deployment/` (mauvais `pyproject.toml`) |
| `Port 8501 is already in use` | Relancer la commande `lsof` ; arrêter l'ancien Streamlit |
| Le navigateur ne s'ouvre pas | Ouvrir <http://localhost:8501> à la main |
