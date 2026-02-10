from typing import Dict, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

_model = None # cache du modele en memoire

def _train_dummy_model() -> Pipeline:
    # Entraine un modele dummy pour l'exemple
    texts = [
        "I love this product!",
        "This is the worst service ever.",
        "Absolutely fantastic experience.",
        "I will never buy this again.",
    ]
    labels = [1, 0, 1, 0]  # 1: positif, 0: negatif

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression())
    ])

    pipeline.fit(texts, labels)
    return pipeline


def load_model() -> Pipeline:
    global _model
    if _model is None:
        _model = _train_dummy_model()
    return _model


def predict(model: Pipeline, text: str) -> int:
    prediction = model.predict_proba([text])[0]
    classes = model.classes_

    best_idx = prediction.argmax()
    label_bool = classes[best_idx]
    confidence = float(prediction[best_idx])

    if label_bool == 1:
        label = "positive"
    else:
        label = "negative"

    return {
        "input_text": text,
        "label": label,
        "confidence": confidence,
        "all_probabilities": {str(cls): float(prob) for cls, prob in zip(classes, prediction)}
    }