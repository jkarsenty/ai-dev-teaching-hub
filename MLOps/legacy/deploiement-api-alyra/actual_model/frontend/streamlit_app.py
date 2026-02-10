import streamlit as st
import requests
from PIL import Image
import io

API_URL = "http://localhost:8000"


st.set_page_config(
    page_title="Sentiment & Image Analyzer",
    layout="centered",
)

st.title("🧠 IA Demo – Sentiment & Image Analysis")
st.write("Ce front Streamlit utilise une API FastAPI pour analyser du texte ou des images.")


# ==== TAB MENU ====
tabs = st.tabs(["📝 Analyse de texte", "🖼 Analyse d'image"])


# ===========================================
#            ANALYSE DE SENTIMENT
# ===========================================
with tabs[0]:
    st.header("Analyse de sentiment")

    user_text = st.text_area("📄 Entrez un texte à analyser :", height=150)

    if st.button("Analyser le sentiment"):
        if not user_text.strip():
            st.warning("Veuillez entrer un texte.")
        else:
            with st.spinner("Analyse en cours..."):
                try:
                    response = requests.post(
                        f"{API_URL}/predict_sentiment",
                        json={"text": user_text},
                    )
                    result = response.json()
                    st.success("Analyse terminée ✔️")
                    st.write(result)

                except Exception as e:
                    st.error(f"Erreur lors de l'appel à l'API : {e}")


# ===========================================
#            ANALYSE D'IMAGE
# ===========================================
with tabs[1]:
    st.header("Analyse d'image")

    uploaded_file = st.file_uploader("📤 Importer une image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Image chargée", use_column_width=True)

    if st.button("Analyser l'image"):
        if uploaded_file is None:
            st.warning("Veuillez importer une image.")
        else:
            with st.spinner("Analyse de l'image..."):
                try:
                    # Convertir l’image en bytes
                    img_bytes = io.BytesIO()
                    img.save(img_bytes, format="PNG")
                    img_bytes.seek(0)

                    files = {"file": ("image.png", img_bytes, "image/png")}

                    response = requests.post(
                        f"{API_URL}/predict_image",
                        files=files
                    )

                    result = response.json()
                    st.success("Analyse terminée ✔️")
                    st.write(result)

                except Exception as e:
                    st.error(f"Erreur lors de l'appel à l'API : {e}")
