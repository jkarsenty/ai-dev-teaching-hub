# README — API de modèle ML/DL

Résumé
- API REST simple pour servir un modèle de Machine Learning / Deep Learning.
- Fournit un endpoint pour prédire à partir de données JSON, un endpoint santé, et exemples d'utilisation.

Principes
- Minimal, reproductible et conteneurisable (Docker).
- Conçu pour être utilisé avec FastAPI / Flask / Flask-RESTX (exemples basés sur FastAPI).

Prérequis
- Python 3.8+
- pip
- Docker (optionnel pour déploiement)
- Modèle sérialisé (ex : model.pt, model.pkl, model.joblib) placé dans ./model/

Installation (locale)
1. Cloner le dépôt ou copier les fichiers dans le répertoire.
2. Créer un environnement virtuel et installer les dépendances :
```bash
python -m venv .venv
source .venv/bin/activate   # ou .venv\Scripts\activate sur Windows
pip install -r requirements.txt
```
3. Placer le fichier du modèle dans le dossier `model/` (nom attendu : `model.pt` ou `model.pkl` selon l'implémentation).

Lancement (locale)
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
- L'API sera disponible sur http://localhost:8000
- Documentation interactive (si FastAPI) : http://localhost:8000/docs

Docker
- Construire l'image :
```bash
docker build -t my-ml-api:latest .
```
- Lancer le conteneur :
```bash
docker run -d --name my-ml-api -p 8000:8000 -v $(pwd)/model:/app/model my-ml-api:latest
```
- Vérifier logs :
```bash
docker logs -f my-ml-api
```

Endpoints (exemples)
- GET /health
    - Description : Vérifie que le service est opérationnel.
    - Réponse : { "status": "ok" }

- POST /predict
    - Description : Retourne la prédiction du modèle pour les données fournies.
    - Content-Type : application/json
    - Exemple de requête :
```json
{
    "features": [5.1, 3.5, 1.4, 0.2]
}
```
    - Exemple curl :
```bash
curl -X POST "http://localhost:8000/predict" \
    -H "Content-Type: application/json" \
    -d '{"features":[5.1,3.5,1.4,0.2]}'
```
    - Exemple de réponse :
```json
{
    "prediction": "setosa",
    "probabilities": [0.98, 0.01, 0.01]
}
```
- POST /predict/batch
    - Accepte une liste d'observations pour traitement par lot.

Bonnes pratiques
- Valider les entrées (pydantic ou schematisation).
- Limiter la taille des payloads pour éviter abus.
- Charger le modèle en mémoire au démarrage (lazy loading possible).
- Ajouter monitoring (metrics, logs) et tests unitaires.

Tests
- Inclure tests unitaires pour la logique de prétraitement et post-traitement du modèle.
- Exemple :
```bash
pytest tests/
```

Déploiement
- Déployer via Docker sur un service cloud (AWS ECS, Azure Container Instances, GCP Run) ou Kubernetes.
- Utiliser une image légère (python:3.8-slim) et multi-stage build si nécessaire.

Fichier attendu dans le repo
- app/ (code API)
- model/model.pt (ou .pkl)
- requirements.txt
- Dockerfile
- README.md
- tests/

Licence et contact
- Licence : MIT (ou préciser la licence du projet)
- Contact : ajouter l'adresse e-mail ou lien vers le repo pour issues

Notes
- Adapter le schéma d'entrée/sortie au type de modèle (classification, régression, segmentation, etc.).
- Documenter clairement les prétraitements appliqués aux features (normalisation, tokenisation...).

Fin.