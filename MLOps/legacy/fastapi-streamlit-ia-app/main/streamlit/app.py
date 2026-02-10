import streamlit as st
import requests
from PIL import Image
import io

st.title("Image Classification App")

def send_image_to_api(image_file):
    files = {
        'file':('2.png', image_file, 'image/png')
    }
    url = 'http://localhost:8000/predict_image'

    try :
        response = requests.post(url, files=files)
        if response.status_code == 200:
            return response.json()
        else:
            return f"Erreur lors de  l'envoi de l'image : {response.status_code} - {response.text}"
    
    except requests.exceptions.RequestException as e:
        return f"Erreur de connexion à l'API : {str(e)}"

with st.form("Image Upload Form"):
    uploaded_file = st.file_uploader("Choisissez une image", type=["png", "jpg", "jpeg"], key='image')
    submit_image = st.form_submit_button("Télécharger l'image")

    if submit_image :
        if uploaded_file is not None:
            image_uploaded = Image.open(uploaded_file)
            st.image(image_uploaded, caption="Image téléchargée", use_column_width=True)
            uploaded_file.seek(0)  # Reset file pointer to the beginning

            response = send_image_to_api(uploaded_file)
            st.write("Réponse de l'API :")
            st.write(response)