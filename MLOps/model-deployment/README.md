# Sentiment Analysis API — Flask Avancé

API REST de sentiment analysis basée sur un pipeline TF-IDF + Régression Logistique, servie avec Flask.

## Lancement

Installer les dépendances :
```bash
pip install -r requirements.txt
```

Lancer l'API :
```bash
python app.py
```

## Endpoints

### `POST /predict`
Prédit le sentiment d'un texte.

**Body JSON :**
```json
{"text": "I love this product"}
```

**Réponse :**
```json
{"text": "I love this product", "sentiment": "positive", "confidence": 0.87}
```

### `GET /health`
Vérifie que l'API tourne.

**Réponse :**
```json
{"status": "ok"}
```