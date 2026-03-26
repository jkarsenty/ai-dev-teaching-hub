# Sentiment Analysis — Streamlit Simple

Interface web de sentiment analysis avec un modèle TF-IDF + LogisticRegression
embarqué directement dans le front, sans API.

## Prérequis
```bash
pip install uv
```

## Lancement local avec uv
```bash
uv venv .venv --python 3.12
source .venv/bin/activate  # Mac / Linux
uv sync
uv run streamlit run app.py
```

L'interface est accessible sur `http://localhost:8501`

## Utilisation

- Entrer un texte dans la zone de saisie
- Cliquer sur **Analyser**
- Le sentiment et la confidence s'affichent directement