# MNIST Classifier — Streamlit Front

Interface web de classification de chiffres manuscrits via un CNN Keras entraîné sur MNIST.
Fonctionne en standalone, sans API.

## Prérequis
```bash
pip install uv
```

## Lancement local avec uv
```bash
uv venv .venv --python 3.12
source .venv/bin/activate  # Mac / Linux
uv sync
uv run python -m streamlit run app.py
```

> Au premier lancement, le modèle est entraîné (~2 min) puis sauvegardé dans `models/`.
> Les lancements suivants chargent directement le modèle sauvegardé.

L'interface est accessible sur `http://localhost:8501`

## Utilisation

### Onglet Dessiner
- Dessinez un chiffre dans la zone noire avec la souris
- Cliquez sur **Prédire**

### Onglet Uploader
- Uploadez une image PNG ou JPEG d'un chiffre manuscrit
- La prédiction s'affiche automatiquement