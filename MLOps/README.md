# MLOps — `mlops/fastapi-simple`

## But de la branche

Même modèle que `flask-avance-uv-docker` (TF-IDF + `LogisticRegression`, réponse
`sentiment` + `confidence`), mais l'API passe de **Flask à FastAPI** :

- schémas **Pydantic** (`PredictRequest`, `PredictResponse`) : validation automatique des entrées ;
- documentation interactive **Swagger** générée sur `/docs` ;
- serveur **`uvicorn`** à la place du serveur de développement Flask ;
- `Dockerfile` adapté (port `8000`, commande `uvicorn`).

Gestion de l'environnement avec **`uv`** (`pyproject.toml`, `uv.lock`, `.python-version`).

## Structure

Voir [`model-deployment/README.md`](model-deployment/README.md).

## Avant de lancer : vérifier les ports

Aucun résultat attendu. Si une ligne s'affiche, le port est déjà pris : arrête le
processus (`kill <PID>`) ou change de port.

```bash
lsof -nP -iTCP:5000,5001,8000,8501 -sTCP:LISTEN
```

> Cette branche utilise le port **8000**.

## Lancer

Voir [`model-deployment/README.md`](model-deployment/README.md).
