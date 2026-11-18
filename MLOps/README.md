# MLOps — `mlops/template-front`

## But de la branche

Deuxième étape de la partie **front** : une interface Streamlit qui embarque un **CNN Keras**
(MNIST), toujours **sans API** :

- deux onglets : **dessiner** un chiffre sur un canvas, ou **uploader** une image PNG/JPEG ;
- le CNN est entraîné au premier lancement puis sauvegardé dans `models/image_model.keras` ;
- dépendances : `streamlit`, `streamlit-drawable-canvas`, `tensorflow`.

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
