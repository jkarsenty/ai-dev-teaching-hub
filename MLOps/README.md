# MLOps — `mlops/template-api`

## But de la branche

Par rapport à `fastapi-avance`, on passe à de **vrais modèles de Deep Learning** et à une
organisation de projet plus professionnelle :

- `POST /predict/text` : sentiment avec **DistilBERT** (`transformers` + `torch`), téléchargé depuis Hugging Face ;
- `POST /predict/image` : chiffres MNIST avec un **CNN Keras** (TensorFlow), plus profond qu'avant ;
- modèles chargés au démarrage via un **`lifespan`** FastAPI (dictionnaire `models`) ;
- poids sauvegardés dans `models/` (dossier non versionné) et rechargés aux lancements suivants.

Environnement géré avec **`uv`** (`pyproject.toml`, `uv.lock`, `.python-version`) et Docker.

> Ressources : `torch` + `tensorflow` sont lourds (plusieurs Go d'installation) ; prévoir
> ≥ 4 Go de RAM libre. Au premier lancement : téléchargement de DistilBERT (~260 Mo,
> internet requis) et entraînement du CNN.

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
