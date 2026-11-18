# MLOps — `mlops/template-front-render`

## But de la branche

Variante de `template-front` prévue pour un **déploiement sur Render** : le front ne contient
plus de modèle, il **appelle l'API** `ml-dl-api` (branche `mlops/template-api-render`) :

- `app.py` envoie l'image en `POST` multipart vers `{API_URL}/predict/image` ;
- `API_URL` est lue dans l'environnement (défaut `http://localhost:8000`) ;
- plus de TensorFlow côté front : `requests` à la place (image beaucoup plus légère) ;
- `Dockerfile` (port `$PORT`), `.dockerignore` et `render.yaml` ajoutés.

Le détail du déploiement est dans
[`model-deployment/DEPLOIEMENT_RENDER.md`](model-deployment/DEPLOIEMENT_RENDER.md).

> ⚠️ Ce front **a besoin de l'API** : elle vit sur une autre branche (`mlops/template-api-render`).

## Structure

Voir [`model-deployment/README.md`](model-deployment/README.md).

## Avant de lancer : vérifier les ports

Aucun résultat attendu. Si une ligne s'affiche, le port est déjà pris : arrête le
processus (`kill <PID>`) ou change de port.

```bash
lsof -nP -iTCP:5000,5001,8000,8501 -sTCP:LISTEN
```

> Cette branche utilise le port **8501** (front) ; l'API appelée utilise le port **8000**.

## Lancer

Voir [`model-deployment/README.md`](model-deployment/README.md).
