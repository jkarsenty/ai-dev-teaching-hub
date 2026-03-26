import os
import numpy as np
from tensorflow import keras

from api.preprocess.image import preprocess_image

MODEL_PATH = "models/image_model.keras"

def create_image_model() -> keras.Model:
    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(32, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=64,
        validation_data=(X_test, y_test),
        verbose=1,
    )
    os.makedirs("models", exist_ok=True)
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
    probas = model.predict(array, verbose=0)[0]
    digit = int(np.argmax(probas))
    confidence = round(float(np.max(probas)), 4)
    return {"digit": digit, "confidence": confidence}