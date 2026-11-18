

## Structure

Voir [`model-deployment/structure_projet.md`](model-deployment/structure_projet.md).

```text
MLOps/
├── README.md                  ← ce fichier
└── model-deployment/
    ├── app.py                 ← API Flask (/predict, /health)
    ├── model.py               ← création / chargement / prédiction
    ├── model.pkl              ← généré au 1er lancement (non versionné)
    └── structure_projet.md
```

> `MLOps/model_deployment/` (avec un underscore) est un squelette à compléter
> par les étudiants : ce n'est pas le projet de cette branche.

## Avant de lancer : vérifier les ports

Aucun résultat attendu. Si une ligne s'affiche, le port est déjà pris : arrête le
processus (`kill <PID>`) ou change de port.

```bash
lsof -nP -iTCP:5000,5001,8000,8501 -sTCP:LISTEN
```

> Sur macOS, le port **5000** est souvent occupé par *AirPlay Receiver*
> (processus `ControlCe`). Désactive-le dans Réglages Système → Général →
> AirDrop et Handoff, ou lance l'API sur 5001.

## Lancer l'API

Se placer dans le dossier du projet (le `model.pkl` est créé dans le dossier courant) :

```bash
cd MLOps/model-deployment
```

Installer les dépendances dans **ton** environnement actif (conda ou venv, jamais le Python global) :

```bash
pip install flask scikit-learn
python app.py
```

L'API écoute sur `http://127.0.0.1:5000`.

## Arrêter l'API

`Ctrl+C` dans le terminal. En mode `debug=True`, Flask démarre un processus fils (reloader) :
si le port reste occupé après l'arrêt, relance la commande `lsof` ci-dessus puis `kill <PID>`.

## Tester

```bash
curl http://127.0.0.1:5000/health

curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product"}'
```

Réponse attendue : `{"text": "I love this product", "sentiment": "positive"}`
(ou `negative` : le modèle est aléatoire). Sans champ `text`, l'API répond `400`.
