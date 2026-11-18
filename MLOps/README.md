# MLOps — `mlops/flask-simple`

## But de la branche

Montrer **l'API la plus simple possible** autour d'un modèle : un fichier `model.py`
(création / chargement / prédiction) et un fichier `app.py` (routes Flask).
Le modèle est un `DummyClassifier` : il répond au hasard, l'objectif est la **mécanique**
(API + modèle sérialisé), pas la qualité de la prédiction.

Volontairement absent sur cette branche : `requirements.txt`, `pyproject.toml`, Docker
(ils arrivent avec `flask-avance` puis `flask-avance-uv-docker`).

## Structure

Voir [`model-deployment/README.md`](model-deployment/README.md).

## Avant de lancer : vérifier les ports

Aucun résultat attendu. Si une ligne s'affiche, le port est déjà pris : arrête le
processus (`kill <PID>`) ou change de port.

```bash
lsof -nP -iTCP:5000,5001,8000,8501 -sTCP:LISTEN
```

> Sur macOS, le port **5000** est souvent occupé par *AirPlay Receiver*
> (processus `ControlCe`). Désactive-le dans Réglages Système → Général →
> AirDrop et Handoff, ou change le port dans `app.py`.

## Lancer

Voir [`model-deployment/README.md`](model-deployment/README.md).
