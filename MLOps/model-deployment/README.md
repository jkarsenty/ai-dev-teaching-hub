# Sentiment Analysis API — Flask avancé

API REST de sentiment analysis (TF-IDF + régression logistique) servie avec Flask.
Environnement géré avec `pip` + `requirements.txt` (venv ou conda).

## Structure du projet

```text
model-deployment/
├── README.md           # ce fichier
├── requirements.txt    # flask, scikit-learn
├── app.py              # API Flask (/predict, /health)
├── model.py            # preprocess, entraînement, chargement, prédiction
└── model.pkl           # généré au 1er lancement (non versionné)
```

| Fichier | Rôle |
|---|---|
| `app.py` | Routes `POST /predict` et `GET /health` |
| `model.py` | `preprocess()`, `create_model()`, `load_model()`, `predict()` ; modèle sérialisé avec `pickle` |
| `requirements.txt` | Dépendances installées avec `pip` |
| `model.pkl` | Modèle entraîné. Supprime-le pour forcer un ré-entraînement |

## Prérequis

1. **Ports libres** (aucun résultat attendu) :
   ```bash
   lsof -nP -iTCP:5000,5001,8000,8501 -sTCP:LISTEN
   ```
   Sur macOS, le port 5000 est parfois pris par *AirPlay Receiver* (`ControlCe`).
2. **Être dans le bon dossier** : toutes les commandes se lancent depuis `MLOps/model-deployment/`
   (le chemin du repo contient des espaces : mets-le entre guillemets).
   ```bash
   cd "<chemin du repo>/MLOps/model-deployment"
   ls app.py model.py requirements.txt     # les 3 fichiers doivent s'afficher
   ```
3. **Python 3.11 ou plus récent** : `python3 --version`.
4. Ne jamais faire `pip install` en dehors d'un environnement activé.

## Installation

### Option A — `venv`

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

### Option B — `conda`

```bash
conda create -n flask-avance python=3.12 -y
conda activate flask-avance
pip install -r requirements.txt
```

Vérifier que l'on utilise bien le bon Python :

```bash
which python        # doit pointer vers .venv/bin/python (ou l'env conda)
```

## Lancer l'API

```bash
python app.py
```

L'API écoute sur `http://127.0.0.1:5000`. Au premier lancement, `model.pkl` est créé.

## Tester

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
