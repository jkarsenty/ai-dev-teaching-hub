# Déploiement sur Render

Cette branche (`mlops/template-api-render`) dérive de `mlops/template-api` avec
les ajustements nécessaires pour déployer l'API sur [Render](https://render.com)
en tant que **Web Service Docker**.

## Ce qui a changé par rapport à `mlops/template-api`

- **`Dockerfile`** :
  - le port d'écoute d'uvicorn utilise désormais la variable `$PORT` fournie
    par Render (`--port ${PORT:-8000}`) au lieu du port `8000` codé en dur ;
  - le modèle image (CNN Keras) est entraîné **pendant le build**
    (`RUN uv run python -c "..."`) plutôt qu'au premier démarrage.
- **`.dockerignore`** (nouveau) : exclut `.venv`, `__pycache__`, les modèles
  locaux (`*.pkl`, `*.keras`) et les artefacts de dev pour ne pas alourdir le
  contexte de build envoyé à Docker.
- **`render.yaml`** (nouveau) : Blueprint Render décrivant le service
  (build Docker, chemin du Dockerfile, healthcheck).
- **Endpoint `/predict/text` (DistilBERT) désactivé** : commenté dans
  `app/main.py` et dans `pyproject.toml` (dépendances `torch`/`transformers`).
  Voir "Tenir sur le plan Free" ci-dessous pour le pourquoi.

## Tenir sur le plan Free (512 Mo)

Un premier essai avec les **deux** modèles actifs (DistilBERT + CNN Keras,
donc torch + tensorflow + transformers chargés simultanément) a échoué sur le
plan Free avec `Out of memory (used over 512Mi)` au démarrage : le simple
*import* de torch et de tensorflow, avant même de charger les poids, dépasse
déjà largement 512 Mo à eux deux.

Options envisagées pour rester sur Free :
1. **Ne garder qu'un seul modèle** (choix retenu ici) — on désactive le texte
   (DistilBERT/torch) et on garde l'image (CNN/tensorflow). torch a un
   baseline mémoire à l'import un peu plus léger que tensorflow, donc le
   supprimer laisse plus de marge — mais rien ne garantit que tensorflow seul
   tienne dans 512 Mo, c'est un pari, pas une certitude.
2. Déployer plutôt la branche `mlops/fastapi-simple` (scikit-learn, sans
   torch/tensorflow) — solution la plus fiable pour rester gratuit.
3. Remplacer les modèles DL par des équivalents scikit-learn.
4. Passer par des runtimes d'inférence allégés (ONNX Runtime, TFLite) au lieu
   des libs complètes torch/tensorflow.

**Pour réactiver `/predict/text`** (nécessite alors un plan Render >= Standard,
2 Go) : décommenter dans `app/main.py` les imports, la ligne dans `lifespan()`
et la route `/predict/text` ; décommenter `torch`/`transformers` dans
`pyproject.toml` ; ajouter la ligne de bake DistilBERT dans le `Dockerfile`.

## Prérequis

- Un compte [Render](https://dashboard.render.com) avec le repo GitHub connecté.
- Docker installé en local pour **tester le build avant de push** :
  ```bash
  cd model-deployment
  docker build -t ml-dl-api .
  docker run -p 8000:8000 -e PORT=8000 ml-dl-api
  ```
- Prévoir un build de quelques minutes : installation de tensorflow (~1 Go de
  dépendances) + entraînement du CNN sur MNIST (~2 min), exécuté pendant le
  build Docker sur les serveurs de Render.

## Différences entre l'exécution locale et le déploiement Render

| | Local (`uv run uvicorn --reload`) | Render (Docker) |
|---|---|---|
| **Port** | fixe, `8000` | dynamique, imposé par `$PORT` (variable d'env injectée par Render) |
| **Filesystem** | persistant sur la machine du dev : `models/` garde les poids d'une exécution à l'autre | **éphémère** — tout ce qui n'est pas dans l'image Docker est perdu à chaque redeploy/restart |
| **Chargement des modèles** | entraîné au premier lancement, réutilisé ensuite depuis `models/` | **baké dans l'image au build** (sinon ré-entraînement à chaque redémarrage → healthcheck en timeout) |
| **Mémoire disponible** | celle de la machine du dev (souvent 8–16 Go) | limitée au plan Render choisi ; le plan Free (512 Mo) est visé ici en gardant un seul modèle (CNN/tensorflow) |
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
