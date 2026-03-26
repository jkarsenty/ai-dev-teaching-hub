import streamlit as st
from model import load_model, predict

st.set_page_config(page_title="Sentiment Analysis", page_icon="💬")

model = load_model()

st.title("💬 Sentiment Analysis")
st.write("Entrez un texte pour analyser son sentiment.")

text = st.text_area("Texte à analyser", placeholder="I love this product...")

if st.button("Analyser"):
    if not text.strip():
        st.warning("Veuillez entrer un texte.")
    else:
        result = predict(model, text)
        sentiment = result["sentiment"]
        confidence = result["confidence"]

        if sentiment == "positive":
            st.success(f"😊 Sentiment : **{sentiment}**")
        else:
            st.error(f"😞 Sentiment : **{sentiment}**")

        st.metric("Confidence", f"{confidence * 100:.1f}%")