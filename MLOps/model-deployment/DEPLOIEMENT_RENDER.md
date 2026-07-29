# Déploiement sur Render

Cette branche (`mlops/template-front-render`) dérive de `mlops/template-front`
et déploie le front Streamlit sur Render en l'appelant l'API `ml-dl-api`
(branche `mlops/template-api-render`) au lieu d'embarquer son propre modèle.

## Ce qui a changé par rapport à `mlops/template-front`

- **`app.py`** : ne charge plus tensorflow/Keras en local. Envoie l'image
  (dessinée ou uploadée) en `POST` multipart vers `{API_URL}/predict/image`
  et affiche la réponse JSON (`digit`, `confidence`). `API_URL` est lu depuis
  une variable d'environnement (défaut `http://localhost:8000`).
- **`pyproject.toml`** : `tensorflow` et `numpy` retirés (plus utilisés côté
  front), `requests` ajouté pour l'appel HTTP.
- **`Dockerfile`** (nouveau) : lance `streamlit run` sur `$PORT` (fourni par
  Render, `8501` par défaut en local).
- **`.dockerignore`** (nouveau).
- **`render.yaml`** (nouveau) : Blueprint avec `API_URL` pointant sur
  `https://ml-dl-api.onrender.com`.

## Pourquoi deux services séparés plutôt qu'un seul

Le front (Streamlit) et l'API (FastAPI/uvicorn) sont deux process serveur
indépendants, chacun avec sa propre boucle d'événements. Render n'expose
qu'**un seul port public par Web Service** : pour les faire tourner tous les
deux derrière une seule URL Render, il faudrait un reverse proxy (nginx/Caddy)
dans le même conteneur, qui route `/` vers Streamlit et `/api/*` vers uvicorn,
avec un superviseur de process (supervisord, honcho...) pour lancer les deux.
C'est faisable mais ajoute de la complexité (config du proxy, gestion des
deux process, un seul plan Render pour la RAM des deux) sans bénéfice réel ici.

**Deux services Free séparés** est plus simple, plus proche de l'usage normal
de Render, et reste à **0 €** au total :
- `ml-dl-api` (branche `mlops/template-api-render`) — le CNN.
- `mnist-front` (cette branche) — l'interface Streamlit, qui appelle l'API
  ci-dessus via HTTP.

## CORS : pas nécessaire ici

L'appel HTTP vers l'API est fait **côté serveur**, dans le process Python de
Streamlit (`requests.post(...)`), pas depuis le navigateur en JavaScript.
Il n'y a donc pas de restriction CORS à gérer côté FastAPI.

> Si un jour ce front est remplacé par une app JS/React qui appelle l'API
> directement depuis le navigateur (cf. branches `mlops/front-simple` /
> `mlops/template-full`), il faudra alors ajouter
> `fastapi.middleware.cors.CORSMiddleware` sur `ml-dl-api` en autorisant
> l'origine du front.

## Prérequis

- Le service `ml-dl-api` déjà déployé (branche `mlops/template-api-render`)
  et son URL Render connue.
- Un compte Render avec le repo GitHub connecté.
- Docker installé en local pour tester le build avant de push :
  ```bash
  cd model-deployment
  docker build -t mnist-front .
  docker run -p 8501:8501 -e PORT=8501 -e API_URL=https://ml-dl-api.onrender.com mnist-front
  ```

## Déploiement

1. Connecter le repo à Render sur la branche `mlops/template-front-render`
   (New → Blueprint en pointant `render.yaml`, ou New → Web Service en
   configurant manuellement Root Directory = `MLOps/model-deployment`).
2. Vérifier/ajuster la variable d'env `API_URL` pour qu'elle pointe sur
   l'URL réelle du service `ml-dl-api` (visible dans son dashboard Render —
   elle peut différer de `ml-dl-api.onrender.com` si ce nom était déjà pris).
3. Plan Free suffisant : Streamlit + `requests` + `pillow`, sans tensorflow,
   a une empreinte mémoire largement sous 512 Mo.
