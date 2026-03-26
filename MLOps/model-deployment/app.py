import streamlit as st
import numpy as np
from PIL import Image
import io
from tensorflow import keras
from streamlit_drawable_canvas import st_canvas

MODEL_PATH = "models/image_model.keras"

st.set_page_config(page_title="MNIST Classifier", page_icon="✏️")

@st.cache_resource
def load_model():
    if not __import__("os").path.exists(MODEL_PATH):
        st.info("Premier lancement : entraînement du modèle en cours (~2 min)...")
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
        model.fit(X_train, y_train, epochs=5, batch_size=64,
                  validation_data=(X_test, y_test), verbose=0)
        __import__("os").makedirs("models", exist_ok=True)
        model.save(MODEL_PATH)
    return keras.models.load_model(MODEL_PATH)

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("L")
    image = image.resize((28, 28))
    array = np.array(image, dtype=np.float32) / 255.0
    array = array.reshape(1, 28, 28, 1)
    return array

model = load_model()

st.title("✏️ MNIST Classifier")
st.write("Dessinez un chiffre ou uploadez une image pour le classifier.")

tab1, tab2 = st.tabs(["✏️ Dessiner", "📁 Uploader"])

with tab1:
    st.write("Dessinez un chiffre dans la zone ci-dessous :")
    canvas = st_canvas(
        fill_color="black",
        stroke_width=20,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    if st.button("Prédire", key="btn_draw"):
        if canvas.image_data is not None:
            image = Image.fromarray(canvas.image_data.astype("uint8"), "RGBA")
            array = preprocess_image(image)
            probas = model.predict(array, verbose=0)[0]
            digit = int(np.argmax(probas))
            confidence = round(float(np.max(probas)), 4)
            st.success(f"Chiffre prédit : **{digit}**")
            st.metric("Confidence", f"{confidence * 100:.1f}%")

with tab2:
    uploaded = st.file_uploader("Uploader une image PNG ou JPEG", type=["png", "jpg", "jpeg"])
    if uploaded:
        image = Image.open(io.BytesIO(uploaded.read()))
        st.image(image, caption="Image uploadée", width=150)
        array = preprocess_image(image)
        probas = model.predict(array, verbose=0)[0]
        digit = int(np.argmax(probas))
        confidence = round(float(np.max(probas)), 4)
        st.success(f"Chiffre prédit : **{digit}**")
        st.metric("Confidence", f"{confidence * 100:.1f}%")