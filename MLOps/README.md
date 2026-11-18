# MLOps — `mlops/fastapi-avance`

## But de la branche

Par rapport à `fastapi-simple` (un fichier `app.py` + un `model.py`), l'API est
**découpée par responsabilité** et expose **deux modèles** :

- `POST /predict/text` : sentiment d'un texte (TF-IDF + `LogisticRegression`) ;
- `POST /predict/image` : chiffre manuscrit MNIST (petit CNN **TensorFlow/Keras**),
  avec les probabilités des 10 classes (`probas`) ;
- dossier `app/` modulaire : `main.py`, `schemas.py`, `models/`, `preprocess/` ;
- exemples d'images dans `examples/` pour tester l'endpoint image.

Environnement géré avec **`uv`** (`pyproject.toml`, `uv.lock`, `.python-version`) et Docker.

> TensorFlow est lourd (plusieurs centaines de Mo) : le premier `uv sync` est long.
> Au premier lancement, MNIST est téléchargé et le CNN entraîné (~30 s).

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
