# Déploiement sur Render

Cette branche (`mlops/template-api-render`) dérive de `mlops/template-api` avec
les ajustements nécessaires pour déployer l'API sur [Render](https://render.com)
en tant que **Web Service Docker**.

## Ce qui a changé par rapport à `mlops/template-api`

- **`Dockerfile`** :
  - le port d'écoute d'uvicorn utilise désormais la variable `$PORT` fournie
    par Render (`--port ${PORT:-8000}`) au lieu du port `8000` codé en dur ;
  - les modèles (DistilBERT + CNN Keras) sont téléchargés/entraînés **pendant
    le build** (`RUN uv run python -c "..."`) plutôt qu'au premier démarrage.
- **`.dockerignore`** (nouveau) : exclut `.venv`, `__pycache__`, les modèles
  locaux (`*.pkl`, `*.keras`) et les artefacts de dev pour ne pas alourdir le
  contexte de build envoyé à Docker.
- **`render.yaml`** (nouveau) : Blueprint Render décrivant le service
  (build Docker, chemin du Dockerfile, healthcheck).

## Prérequis

- Un compte [Render](https://dashboard.render.com) avec le repo GitHub connecté.
- Un plan payant **Standard (2 Go de RAM) minimum**. Le plan Free (512 Mo) est
  insuffisant : torch, tensorflow et transformers chargés simultanément
  dépassent largement cette limite.
- Docker installé en local pour **tester le build avant de push** :
  ```bash
  cd model-deployment
  docker build -t ml-dl-api .
  docker run -p 8000:8000 -e PORT=8000 ml-dl-api
  ```
- Prévoir un build long (plusieurs minutes) : installation de torch/tensorflow
  (~1–2 Go de dépendances) + téléchargement des poids DistilBERT (~260 Mo) +
  entraînement du CNN sur MNIST (~2 min), le tout exécuté pendant le build
  Docker sur les serveurs de Render.

## Différences entre l'exécution locale et le déploiement Render

| | Local (`uv run uvicorn --reload`) | Render (Docker) |
|---|---|---|
| **Port** | fixe, `8000` | dynamique, imposé par `$PORT` (variable d'env injectée par Render) |
| **Filesystem** | persistant sur la machine du dev : `models/` garde les poids d'une exécution à l'autre | **éphémère** — tout ce qui n'est pas dans l'image Docker est perdu à chaque redeploy/restart |
| **Chargement des modèles** | téléchargés/entraînés au premier lancement, réutilisés ensuite depuis `models/` | **bakés dans l'image au build** (sinon re-téléchargement + ré-entraînement à chaque redémarrage → healthcheck en timeout) |
| **Mémoire disponible** | celle de la machine du dev (souvent 8–16 Go) | limitée au plan Render choisi ; nécessite Standard (2 Go) mini pour torch+tensorflow+transformers en simultané |
| **Reload à chaud** | `--reload` activable pour itérer vite | jamais utilisé en production |
| **Logs** | dans le terminal du dev | agrégés automatiquement dans le dashboard Render (stdout/stderr) |
| **Mise à jour du code** | immédiate (fichiers locaux) | nécessite un nouveau build/déploiement (push sur la branche connectée) |

## Alternative envisagée : disque persistant Render

Au lieu de baker les modèles dans l'image (build plus long, image plus
lourde), on pourrait attacher un [Render Disk](https://render.com/docs/disks)
persistant monté sur `models/` et laisser l'appli télécharger/entraîner au
premier démarrage, comme en local. C'est un coût supplémentaire et un premier
démarrage lent (cold start ~2-3 min) ; l'approche "bake au build" a été
préférée ici pour un démarrage instantané et un comportement reproductible.

## Déploiement

1. Connecter le repo à Render (New → Blueprint, ou New → Web Service en
   pointant `render.yaml`).
2. Vérifier que le **Root/Build context** correspond bien à `model-deployment/`.
3. Le healthcheck est configuré sur `GET /health`.
4. Premier déploiement = premier build long (voir ci-dessus) ; les suivants
   ne re-téléchargent/ré-entraînent que si le Dockerfile ou les dépendances
   changent (cache de layers Docker).
