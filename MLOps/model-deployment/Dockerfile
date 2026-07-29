FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml .

RUN uv sync --no-dev

COPY app/ ./app/

RUN mkdir -p models

# Render a un filesystem éphémère : on entraîne le CNN ici, au build, pour
# qu'il soit déjà présent dans l'image au démarrage (sinon chaque
# redeploy/restart le ré-entraînerait, ce qui ferait timeout le healthcheck).
# Le modèle texte (DistilBERT) est désactivé, voir app/main.py.
RUN uv run python -c "from app.models.image_model import load_image_model; load_image_model()"

EXPOSE 8000

# Render fournit le port via la variable $PORT ; on retombe sur 8000 en local.
CMD ["sh", "-c", "uv run uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]