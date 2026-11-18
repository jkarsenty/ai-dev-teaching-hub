# MLOps — `mlops/flask-avance-uv-docker`

## But de la branche

Même API et même modèle que `flask-avance` (TF-IDF + `LogisticRegression`, réponse
`sentiment` + `confidence`). Ce qui change :

- **`uv`** remplace `pip` + `requirements.txt` : `pyproject.toml` (dépendances) et
  `uv.lock` (versions figées), `.python-version` (Python 3.11) ;
- un **`Dockerfile`** pour lancer l'API dans un conteneur.

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
> AirDrop et Handoff, ou lance le conteneur sur un autre port (`-p 5001:5000`).

## Lancer

Voir [`model-deployment/README.md`](model-deployment/README.md).
