# 3. Exemple prêt à adapter

Hypothèses : projet Python géré avec `uv`, tests avec `pytest`, lint avec `ruff`, image Docker.
Adapte les chemins (ex : `working-directory`) à ton projet.

## `ci.yml` : lint + tests

```yaml
name: CI

on:
  pull_request:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: MLOps/model-deployment   # dossier du projet
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --frozen          # installe selon uv.lock
      - run: uv run ruff check .       # lint
      - run: uv run pytest             # tests
```

## `cd.yml` : build + publication de l'image

```yaml
name: CD

on:
  push:
    branches: [main]

jobs:
  docker:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}
      - uses: docker/build-push-action@v6
        with:
          context: MLOps/model-deployment
          push: true
          tags: moncompte/sentiment-api:latest
```

> Le déploiement final dépend de l'hébergeur (Render, serveur, etc.) : souvent un *deploy hook* ou une
> connexion directe au repo, à configurer côté hébergeur.

## Mise en place pas à pas

| Étape | Action |
|-------|--------|
| 1 | Vérifier que le projet marche en local (`uv sync`, `uv run pytest`) |
| 2 | Créer `.github/workflows/ci.yml` à la **racine du repo** |
| 3 | Commit + push sur une branche `feat/ci` |
| 4 | Ouvrir une Pull Request, regarder l'onglet **Actions** |
| 5 | Corriger jusqu'au ✓ vert |
| 6 | Ajouter les secrets, puis `cd.yml` |
| 7 | Dans GitHub → Settings → Branches : exiger la CI verte avant de merger sur `main` |

## Erreurs fréquentes

| Symptôme | Cause probable |
|----------|----------------|
| `No such file` | `working-directory` incorrect ou workflow hors de `.github/workflows/` |
| Le workflow ne se lance pas | Mauvais `on:` ou mauvaise branche |
| ✗ sur `uv sync --frozen` | `uv.lock` non commité ou pas à jour |
| ✗ sur `docker login` | Secret absent ou mal nommé |
