FROM python:3.11-slim

# Installer uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copier les fichiers de dépendances en premier (cache Docker)
COPY pyproject.toml .

# Installer les dépendances
RUN uv sync --no-dev

# Copier le reste du code
COPY app.py model.py ./

EXPOSE 5000

CMD ["uv", "run", "python", "app.py"]
