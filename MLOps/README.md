# MLOps — `mlops/flask-avance`

## But de la branche

Par rapport à `flask-simple`, on passe d'un modèle aléatoire à un **vrai modèle de
classification de texte** (TF-IDF + `LogisticRegression`) avec :

- une fonction `preprocess()` (minuscules, ponctuation, espaces) ;
- une réponse enrichie : `sentiment` **et** `confidence` ;
- un `requirements.txt` et un `README` dédié dans `model-deployment/`.

Le projet est géré avec **`pip`** dans un environnement `venv` ou `conda` que tu crées
toi-même : il n'y a donc **pas de `pyproject.toml`** dans `model-deployment/`
(il arrive avec `flask-avance-uv-docker`).

## Structure

La structure détaillée est dans [`model-deployment/README.md`](model-deployment/README.md).

```text
MLOps/
├── README.md                  ← ce fichier
└── model-deployment/
    ├── README.md              ← structure, installation, commandes
    ├── requirements.txt
    ├── app.py
    └── model.py
```

## Avant de lancer : vérifier les ports

Aucun résultat attendu. Si une ligne s'affiche, le port est déjà pris : arrête le
processus (`kill <PID>`) ou change de port.

```bash
lsof -nP -iTCP:5000,5001,8000,8501 -sTCP:LISTEN
```

> Sur macOS, le port **5000** est souvent occupé par *AirPlay Receiver*
> (processus `ControlCe`). Désactive-le dans Réglages Système → Général →
> AirDrop et Handoff, ou change le port dans `app.py`.

## Lancer

Voir [`model-deployment/README.md`](model-deployment/README.md).
