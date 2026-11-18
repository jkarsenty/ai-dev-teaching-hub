# MLOps — `mlops/template-full`

## But de la branche

Dernière étape : **API + front dans la même branche**, qui communiquent en HTTP.

- `api/` : l'API FastAPI de `template-api` (DistilBERT pour le texte, CNN pour les chiffres) ;
- `front/app.py` : interface Streamlit à deux onglets (sentiment de texte, chiffres MNIST) qui appelle l'API ;
- `docker-compose.yml` : lance les deux services ; `./models` est monté dans le conteneur de l'API.

Environnement géré avec **`uv`** (un seul `pyproject.toml` pour les deux services).

> Ressources : `torch` + `tensorflow` + `streamlit` dans le même environnement ; prévoir
> ≥ 4 Go de RAM libre. Au premier lancement : DistilBERT (~260 Mo) + entraînement du CNN.

## Structure

Voir [`model-deployment/README.md`](model-deployment/README.md).

## Avant de lancer : vérifier les ports

Aucun résultat attendu. Si une ligne s'affiche, le port est déjà pris : arrête le
processus (`kill <PID>`) ou change de port.

```bash
lsof -nP -iTCP:5000,5001,8000,8501 -sTCP:LISTEN
```

> Cette branche utilise les ports **8000** (API) et **8501** (front).

## Lancer

Voir [`model-deployment/README.md`](model-deployment/README.md).
