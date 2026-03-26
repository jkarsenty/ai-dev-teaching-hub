import streamlit as st
import requests
import io
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import numpy as np

API_URL = "http://localhost:8000"

st.set_page_config(page_title="ML & DL App", page_icon="🤖")
st.title("🤖 ML & DL App")

tab1, tab2 = st.tabs(["💬 Sentiment Analysis", "✏️ MNIST Classifier"])

# ── Onglet Texte ──────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Analyse de sentiment")
    st.write("Entrez un texte pour analyser son sentiment via l'API.")

    text = st.text_area("Texte à analyser", placeholder="I love this product...")

    if st.button("Analyser", key="btn_text"):
        if not text.strip():
            st.warning("Veuillez entrer un texte.")
        else:
            try:
                response = requests.post(
                    f"{API_URL}/predict/text",
                    json={"text": text},
                )
                response.raise_for_status()
                data = response.json()
                sentiment = data["sentiment"]
                confidence = data["confidence"]

                if sentiment == "positive":
                    st.success(f"😊 Sentiment : **{sentiment}**")
                else:
                    st.error(f"😞 Sentiment : **{sentiment}**")
                st.metric("Confidence", f"{confidence * 100:.1f}%")

            except requests.exceptions.ConnectionError:
                st.error("❌ Impossible de contacter l'API. Vérifiez qu'elle tourne sur le port 8000.")
            except requests.exceptions.HTTPError as e:
                st.error(f"❌ Erreur API : {e}")

# ── Onglet Image ──────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Classification de chiffres manuscrits")
    st.write("Dessinez un chiffre ou uploadez une image.")

    tab_draw, tab_upload = st.tabs(["✏️ Dessiner", "📁 Uploader"])

    with tab_draw:
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
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                buf.seek(0)
                try:
                    response = requests.post(
                        f"{API_URL}/predict/image",
                        files={"file": ("drawing.png", buf, "image/png")},
                    )
                    response.raise_for_status()
                    data = response.json()
                    st.success(f"Chiffre prédit : **{data['digit']}**")
                    st.metric("Confidence", f"{data['confidence'] * 100:.1f}%")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Impossible de contacter l'API. Vérifiez qu'elle tourne sur le port 8000.")
                except requests.exceptions.HTTPError as e:
                    st.error(f"❌ Erreur API : {e}")

    with tab_upload:
        uploaded = st.file_uploader("Uploader une image PNG ou JPEG", type=["png", "jpg", "jpeg"])
        if uploaded:
            image = Image.open(io.BytesIO(uploaded.read()))
            st.image(image, caption="Image uploadée", width=150)
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            buf.seek(0)
            try:
                response = requests.post(
                    f"{API_URL}/predict/image",
                    files={"file": ("image.png", buf, "image/png")},
                )
                response.raise_for_status()
                data = response.json()
                st.success(f"Chiffre prédit : **{data['digit']}**")
                st.metric("Confidence", f"{data['confidence'] * 100:.1f}%")
            except requests.exceptions.ConnectionError:
                st.error("❌ Impossible de contacter l'API. Vérifiez qu'elle tourne sur le port 8000.")
            except requests.exceptions.HTTPError as e:
                st.error(f"❌ Erreur API : {e}")