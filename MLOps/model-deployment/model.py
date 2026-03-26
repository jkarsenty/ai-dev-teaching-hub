import os
import pickle
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

MODEL_PATH = "model.pkl"

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

def preprocess(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def create_model() -> Pipeline:
    texts, labels = zip(*TRAIN_DATA)
    texts = [preprocess(t) for t in texts]
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression()),
    ])
    pipeline.fit(texts, labels)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print("Modèle créé et sauvegardé.")
    return pipeline

def load_model() -> Pipeline:
    if not os.path.exists(MODEL_PATH):
        print("Aucun modèle trouvé, création en cours...")
        return create_model()
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Modèle chargé.")
    return model

def predict(model: Pipeline, text: str) -> dict:
    cleaned = preprocess(text)
    label = model.predict([cleaned])[0]
    proba = model.predict_proba([cleaned])[0]
    classes = model.classes_.tolist()
    confidence = round(float(max(proba)), 4)
    return {"sentiment": label, "confidence": confidence}