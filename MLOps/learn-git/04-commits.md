# 4. Les messages de commit

## Conventional Commits (simplifié)

Format :

```
type(portée optionnelle): description courte à l'impératif
```

| Type | Quand l'utiliser | Exemple |
|------|------------------|---------|
| `feat` | Nouvelle fonctionnalité | `feat: ajoute la route /predict` |
| `fix` | Correction de bug | `fix: corrige l'erreur de port` |
| `docs` | Documentation seule | `docs: ajoute le guide Git` |
| `refactor` | Réécriture sans changer le comportement | `refactor: simplifie le chargement du modèle` |
| `test` | Ajout / modif de tests | `test: couvre le endpoint /health` |
| `chore` | Maintenance, config, dépendances | `chore: met à jour uv.lock` |

Avec portée : `fix(api): gère le modèle manquant`.

## Bonnes pratiques

| ✅ À faire | ❌ À éviter |
|-----------|------------|
| Un commit = **une** idée | Un énorme commit « tout » |
| Message clair, ~50 caractères | `update`, `fix`, `ok`, `wip` |
| Verbe à l'impératif (`ajoute`, `corrige`) | Phrases vagues |
| Commiter souvent, par petits pas | Commiter une fois par semaine |
| Ne jamais commiter de secrets (`.env`, clés) | Mots de passe dans le code |

## Changement incompatible
Ajoute `!` après le type : `feat!: change le format de la réponse API`.
