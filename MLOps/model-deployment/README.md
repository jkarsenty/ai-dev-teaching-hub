# MNIST Classifier — Streamlit Front

Interface web de classification de chiffres manuscrits via un CNN Keras entraîné sur MNIST.
Fonctionne en standalone, sans API. Environnement géré avec `uv`.

## Structure du projet

```text
model-deployment/
├── README.md            # ce fichier
├── pyproject.toml       # streamlit, streamlit-drawable-canvas, tensorflow, pillow, numpy
├── uv.lock              # versions figées (à committer)
├── .python-version      # Python utilisé par uv (3.12)
├── app.py               # interface Streamlit + CNN (entraînement, chargement, prédiction)
├── model.py             # hérité de front-simple (sentiment), non utilisé ici
├── models/              # CNN sauvegardé (contenu non versionné, .gitkeep)
├── examples/examples/     # images de test (2.png, 4.png, 5.png, mnist_examples.png)
└── .venv/               # créé par uv (non versionné)
```

| Fichier | Rôle |
|---|---|
| `app.py` | Interface (onglets Dessiner / Uploader), entraînement puis chargement du CNN |
| `models/image_model.keras` | CNN sauvegardé au premier lancement |
| `examples/` | Images pour tester l'onglet Uploader |

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
   > et créerait un `.venv` à la racine. Le CNN est sauvegardé dans `models/` **du dossier courant**.
4. **Connexion internet au premier lancement** (téléchargement de MNIST, ~11 Mo).

## Lancement local avec uv

```bash
uv sync          # crée .venv et installe les dépendances
uv run python -m streamlit run app.py
```

- `uv sync` **crée lui-même** le `.venv` : pas besoin de `uv venv` ni de `source activate`.
- Python 3.12 est choisi via `.python-version` (uv le télécharge s'il est absent).
- Au premier lancement, le CNN est entraîné (~1 min) puis sauvegardé dans `models/` ;
  les lancements suivants le rechargent directement.
- Au premier lancement, Streamlit peut demander un e-mail : appuie sur Entrée pour passer.

Vérifier le venv :

```bash
uv run python -c "import sys; print(sys.executable)"
# → .../MLOps/model-deployment/.venv/bin/python
```

L'application écoute sur `http://localhost:8501`.

## Tester

1. Ouvrir <http://localhost:8501> (attendre la fin de l'entraînement au premier lancement).
2. Onglet **Dessiner** : tracer un chiffre dans la zone noire, puis **Prédire**.
3. Onglet **Uploader** : choisir `examples/examples/2.png` (ou `4.png`, `5.png`) ; la prédiction
   s'affiche automatiquement avec la confiance.

```bash
curl http://localhost:8501/_stcore/health     # → ok
```

## Arrêter l'application

`Ctrl+C` dans le terminal. Si le port 8501 reste occupé, relance la commande `lsof`
ci-dessus puis `kill <PID>`.

## Dépannage

| Symptôme | Cause probable |
|---|---|
| `ModuleNotFoundError: streamlit` | Commande lancée hors de `model-deployment/` |
| Le CNN est ré-entraîné à chaque fois | Lancement depuis un autre dossier (`models/` introuvable) |
| Le canvas de dessin ne s'affiche pas | Incompatibilité possible `streamlit-drawable-canvas` / version de Streamlit : utiliser l'onglet **Uploader** |
| `Port 8501 is already in use` | Relancer la commande `lsof` ; arrêter l'ancien Streamlit |
