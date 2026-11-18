# MLOps — `mlops/front-simple`

## But de la branche

Première étape de la partie **front** : une interface **Streamlit** qui embarque son propre modèle
(TF-IDF + `LogisticRegression`), **sans API** :

- `app.py` : l'interface (zone de texte, bouton « Analyser », sentiment + confiance) ;
- `model.py` : le modèle, identique à celui des branches Flask / FastAPI ;
- un seul processus, une seule commande : `uv run streamlit run app.py`.

Environnement géré avec **`uv`** (`pyproject.toml`, `uv.lock`, `.python-version`). Pas de Docker ici.

## Structure

Voir [`model-deployment/README.md`](model-deployment/README.md).

## Avant de lancer : vérifier les ports

Aucun résultat attendu. Si une ligne s'affiche, le port est déjà pris : arrête le
processus (`kill <PID>`) ou change de port.

```bash
lsof -nP -iTCP:5000,5001,8000,8501 -sTCP:LISTEN
```

> Cette branche utilise le port **8501** (Streamlit).

## Lancer

Voir [`model-deployment/README.md`](model-deployment/README.md).
