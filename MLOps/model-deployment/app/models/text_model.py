import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from app.preprocess.text import preprocess_text

MODEL_PATH = "text_model.pkl"

TRAIN_DATA = [
    ("I love this product, it is amazing", "positive"),
    ("Absolutely wonderful experience", "positive"),
    ("This is the best thing I have ever bought", "positive"),
    ("Really happy with the quality", "positive"),
    ("Fantastic service and fast delivery", "positive"),
    ("I hate this product, it is terrible", "negative"),
    ("Worst experience of my life", "negative"),
    ("Completely disappointed, do not buy", "negative"),
    ("Very bad quality, broke after one day", "negative"),
    ("Awful service, never coming back", "negative"),
]

def create_text_model() -> Pipeline:
    texts, labels = zip(*TRAIN_DATA)
    texts = [preprocess_text(t) for t in texts]
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression()),
    ])
    pipeline.fit(texts, labels)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print("Modèle texte créé et sauvegardé.")
    return pipeline

def load_text_model() -> Pipeline:
    if not os.path.exists(MODEL_PATH):
        print("Aucun modèle texte trouvé, création en cours...")
        return create_text_model()
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Modèle texte chargé.")
    return model

def predict_text(model: Pipeline, text: str) -> dict:
    cleaned = preprocess_text(text)
    label = model.predict([cleaned])[0]
    proba = model.predict_proba([cleaned])[0]
    confidence = round(float(max(proba)), 4)
    return {"sentiment": label, "confidence": confidence}