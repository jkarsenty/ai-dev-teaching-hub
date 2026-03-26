import os
import numpy as np
from tensorflow import keras
from app.preprocess.image import preprocess_image

MODEL_PATH = "image_model.keras"

def create_image_model() -> keras.Model:
    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(32, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    (X_train, y_train), _ = keras.datasets.mnist.load_data()
    X_train = X_train.astype("float32") / 255.0
    X_train = X_train.reshape(-1, 28, 28, 1)
    model.fit(X_train, y_train, epochs=3, batch_size=64, verbose=1)
    model.save(MODEL_PATH)
    print("Modèle image créé et sauvegardé.")
    return model

def load_image_model() -> keras.Model:
    if not os.path.exists(MODEL_PATH):
        print("Aucun modèle image trouvé, création en cours...")
        return create_image_model()
    model = keras.models.load_model(MODEL_PATH)
    print("Modèle image chargé.")
    return model

def predict_image(model: keras.Model, image_bytes: bytes) -> dict:
    array = preprocess_image(image_bytes)
    probas = model.predict(array)[0]
    digit = int(np.argmax(probas))
    confidence = round(float(np.max(probas)), 4)
    return {"digit": digit, "confidence": confidence}