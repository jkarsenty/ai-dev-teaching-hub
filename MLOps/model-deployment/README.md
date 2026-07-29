# MNIST Classifier — Streamlit Front

Interface web de classification de chiffres manuscrits. Appelle l'API
`ml-dl-api` (branche `mlops/template-api-render`) via `POST /predict/image` —
voir [DEPLOIEMENT_RENDER.md](./DEPLOIEMENT_RENDER.md).

## Prérequis
```bash
pip install uv
```

## Lancement local avec uv

Lancer d'abord l'API (`model-deployment` de `mlops/template-api-render`) sur
`http://localhost:8000`, puis :
```bash
uv venv .venv --python 3.12
source .venv/bin/activate  # Mac / Linux
uv sync
API_URL=http://localhost:8000 uv run python -m streamlit run app.py
```

Sans `API_URL`, l'app pointe par défaut sur `http://localhost:8000`.

L'interface est accessible sur `http://localhost:8501`

## Utilisation

### Onglet Dessiner
- Dessinez un chiffre dans la zone noire avec la souris
- Cliquez sur **Prédire**

### Onglet Uploader
- Uploadez une image PNG ou JPEG d'un chiffre manuscrit
- La prédiction s'affiche automatiquement