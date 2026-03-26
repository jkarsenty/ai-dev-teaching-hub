# MLOps — Model Deployment

Ce dossier contient le code relatif au déploiement de modèles de Machine Learning et de Deep Learning sous forme d'API et d'interfaces web.

Tout le code se trouve dans le dossier `model-deployment/` dont le contenu **change selon la branche Git active**.

## Comment naviguer entre les branches
```bash
# Voir toutes les branches disponibles
git branch -a

# Changer de branche
git checkout mlops/flask-simple
cd MLOps/model-deployment
```

---

## Vue d'ensemble des branches

### 🔵 Partie Flask — ML classique

| Branche | Modèle | Points clés |
|---|---|---|
| `mlops/flask-simple` | Dummy model | Structure minimale `app.py` + `model.py`, Flask, pickle |
| `mlops/flask-avancé` | TF-IDF + LogisticRegression | Ajout de `preprocess()`, `requirements.txt`, README |
| `mlops/flask-avancé-uv-docker` | TF-IDF + LogisticRegression | `pyproject.toml`, gestionnaire `uv`, Dockerfile |

### 🟣 Partie FastAPI — ML puis DL

| Branche | Modèle | Points clés |
|---|---|---|
| `mlops/fastapi-simple` | TF-IDF + LogisticRegression | Migration Flask → FastAPI, adaptation Dockerfile |
| `mlops/fastapi-avancé` | TF-IDF (texte) + CNN (MNIST) | Dossier `app/` modulaire, fichiers séparés par responsabilité |
| `mlops/template-api` | BERT (texte) + CNN Keras (MNIST) | Vrai modèle DL, TensorFlow/Keras, endpoints multiples, organisation professionnelle |

### 🟢 Partie Front — Streamlit

| Branche | Modèle | Points clés |
|---|---|---|
| `mlops/front-simple` | TF-IDF + LogisticRegression | Streamlit standalone, sans API, modèle ML embarqué |
| `mlops/front-template` | CNN Keras (MNIST) | Streamlit standalone, sans API, upload ou dessin de chiffre |
| `mlops/template-full` | BERT (texte) + CNN Keras (MNIST) | Stack complète : API FastAPI + front Streamlit, deux onglets |

---

## Progression pédagogique
```
flask-simple
    │  + preprocess, vrai modèle ML, requirements
    ▼
flask-avancé
    │  + uv, pyproject.toml, Docker
    ▼
flask-avancé-uv-docker
    │  + migration FastAPI
    ▼
fastapi-simple
    │  + structure modulaire, MNIST
    ▼
fastapi-avancé
    │  + vrai modèle DL (BERT + CNN), organisation pro
    ▼
template-api
    │  + interface Streamlit (sans API)
    ▼
front-simple → front-template
    │  + connexion API ↔ Front
    ▼
template-full
```

---

## Stack technique globale

| Outil | Rôle |
|---|---|
| `scikit-learn` | Modèles ML (TF-IDF, LogisticRegression) |
| `tensorflow` / `keras` | Modèles DL (CNN, BERT) |
| `flask` | API REST — introduction |
| `fastapi` | API REST — production |
| `uv` | Gestionnaire de dépendances moderne |
| `docker` | Conteneurisation |
| `streamlit` | Interface web front-end |