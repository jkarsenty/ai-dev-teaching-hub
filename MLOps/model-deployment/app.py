import io
import os
import requests
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="MNIST Classifier", page_icon="✏️")

def predict_via_api(image: Image.Image) -> dict:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    buf.seek(0)
    response = requests.post(
        f"{API_URL}/predict/image",
        files={"file": ("digit.png", buf, "image/png")},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()

def show_prediction(image: Image.Image) -> None:
    try:
        result = predict_via_api(image)
    except requests.RequestException as e:
        st.error(f"Erreur en appelant l'API ({API_URL}) : {e}")
        return
    st.success(f"Chiffre prédit : **{result['digit']}**")
    st.metric("Confidence", f"{result['confidence'] * 100:.1f}%")

st.title("✏️ MNIST Classifier")
st.write("Dessinez un chiffre ou uploadez une image pour le classifier.")
st.caption(f"API : {API_URL}")

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
            show_prediction(image)

with tab2:
    uploaded = st.file_uploader("Uploader une image PNG ou JPEG", type=["png", "jpg", "jpeg"])
    if uploaded:
        image = Image.open(io.BytesIO(uploaded.read()))
        st.image(image, caption="Image uploadée", width=150)
        show_prediction(image)