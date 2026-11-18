# MLOps — `mlops/template-api-render`

## But de la branche

Variante de `template-api` prévue pour un **déploiement sur Render** :

- seul `POST /predict/image` (CNN Keras) est actif : `POST /predict/text` (DistilBERT) est
  **désactivé**, torch + transformers dépassant les 512 Mo du plan Free ;
- le CNN est **entraîné pendant le build Docker** (le filesystem de Render est éphémère) ;
- le port d'écoute vient de la variable `$PORT` fournie par Render (8000 par défaut en local) ;
- `render.yaml` (Blueprint) et `.dockerignore` ajoutés.

Le détail du déploiement (choix, limites mémoire, étapes) est dans
[`model-deployment/DEPLOIEMENT_RENDER.md`](model-deployment/DEPLOIEMENT_RENDER.md).

## Structure

Voir [`model-deployment/README.md`](model-deployment/README.md).

## Avant de lancer : vérifier les ports

Aucun résultat attendu. Si une ligne s'affiche, le port est déjà pris : arrête le
processus (`kill <PID>`) ou change de port.

```bash
lsof -nP -iTCP:5000,5001,8000,8501 -sTCP:LISTEN
```

> Cette branche utilise le port **8000** en local.

## Lancer

Voir [`model-deployment/README.md`](model-deployment/README.md).
