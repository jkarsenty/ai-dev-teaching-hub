import os
import pickle
import numpy as np
from sklearn.dummy import DummyClassifier

MODEL_PATH = "model.pkl"

def create_model():
    X = np.zeros((10, 1))
    y = ["positive", "negative", "positive", "negative", "positive",
         "negative", "positive", "negative", "positive", "negative"]
    model = DummyClassifier(strategy="uniform")
    model.fit(X, y)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print("Modèle créé et sauvegardé.")
    return model

def load_model():
    if not os.path.exists(MODEL_PATH):
        print("Aucun modèle trouvé, création en cours...")
        return create_model()
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Modèle chargé.")
    return model

def predict(model, text: str) -> str:
    result = model.predict(np.zeros((1, 1)))
    return result[0]