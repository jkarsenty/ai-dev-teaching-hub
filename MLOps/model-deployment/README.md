# Sentiment API — Flask simple

API REST minimale autour d'un modèle (`DummyClassifier`, réponses aléatoires) servie avec Flask.
Pas de gestion de dépendances ni de Docker sur cette branche.

## Structure du projet

```text
model-deployment/
├── README.md    # ce fichier
├── app.py       # API Flask (/predict, /health)
├── model.py     # création / chargement / prédiction
└── model.pkl    # généré au 1er lancement (non versionné)
```

| Fichier | Rôle |
|---|---|
| `app.py` | Routes `POST /predict` et `GET /health` |
| `model.py` | `create_model()`, `load_model()`, `predict()` ; modèle sérialisé avec `pickle` |
| `model.pkl` | Modèle sauvegardé. Supprime-le pour le recréer |

## Prérequis

1. **Ports libres** (aucun résultat attendu) :
   ```bash
   lsof -nP -iTCP:5000,5001,8000,8501 -sTCP:LISTEN
   ```
   Sur macOS, le port 5000 est parfois pris par *AirPlay Receiver* (`ControlCe`).

2. **Être dans le bon dossier** : `model.pkl` est créé dans le dossier courant. Le chemin du
   repo contient des espaces : mets-le entre guillemets.
   ```bash
   cd "<chemin du repo>/MLOps/model-deployment"
   ls app.py model.py     # les 2 fichiers doivent s'afficher
   ```
3. **Un environnement Python actif** (venv ou conda, jamais le Python global).

## Installation

```bash
pip install flask scikit-learn
```

## Lancer l'API

```bash
python app.py
```

L'API écoute sur `http://127.0.0.1:5000`.

## Tester

```bash
curl http://127.0.0.1:5000/health

curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product"}'
```

Réponse (`negative` possible : le modèle est aléatoire) :

```json
{"text": "I love this product", "sentiment": "positive"}
```

Sans champ `text`, l'API répond `400`.

## Arrêter l'API

`Ctrl+C` dans le terminal. En mode `debug=True`, Flask démarre un processus fils (reloader) :
si le port reste occupé après l'arrêt, relance la commande `lsof` ci-dessus puis `kill <PID>`.

## Endpoints

| Méthode | Route | Corps | Réponse |
|---|---|---|---|
| `GET` | `/health` | — | `{"status": "ok"}` |
| `POST` | `/predict` | `{"text": "..."}` | `{"text", "sentiment"}` |
