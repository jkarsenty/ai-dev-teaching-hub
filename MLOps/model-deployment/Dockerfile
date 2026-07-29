FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml .

RUN uv sync --no-dev

COPY app.py .

EXPOSE 8501

# Render fournit le port via $PORT ; on retombe sur 8501 en local.
CMD ["sh", "-c", "uv run streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true"]
