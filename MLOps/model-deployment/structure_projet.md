# Structure du projet — `mlops/flask-simple`

Branche **sans README dédié** dans `model-deployment/` : ce fichier décrit son contenu.
Prérequis et commandes de lancement : [`../README.md`](../README.md).

## Arborescence

```text
model-deployment/
├── app.py                 # API Flask
├── model.py               # modèle (DummyClassifier)
├── model.pkl              # généré au premier lancement (non versionné)
└── structure_projet.md    # ce fichier
```

## Rôle des fichiers

| Fichier | Rôle |
|---|---|
| `app.py` | Routes Flask : `POST /predict`, `GET /health` |
| `model.py` | `create_model()`, `load_model()`, `predict()` ; sérialisation avec `pickle` |
| `model.pkl` | Modèle sauvegardé. Créé automatiquement s'il est absent ; supprime-le pour le recréer |

## Endpoints

| Méthode | Route | Corps | Réponse |
|---|---|---|---|
| `GET` | `/health` | — | `{"status": "ok"}` |
| `POST` | `/predict` | `{"text": "..."}` | `{"text": "...", "sentiment": "positive"}` |

## Ce que cette branche apporte

L'API la plus simple possible autour d'un modèle, sans gestion de dépendances.
