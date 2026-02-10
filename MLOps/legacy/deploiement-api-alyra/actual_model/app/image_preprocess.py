# import des librairies
import numpy as np
from PIL import Image
import io

# fonction de preprocessing des données d'entrée
def  preprocess_image(image):
    # Convertir l'image en tableau numpy
    image = Image.open(io.BytesIO(image))
    
    # Redimensionner l'image à la taille attendue par le modèle
    image = image.resize((28, 28))  # redimensionnement similaire aux modele
    
    # convertion en noir et blanc
    img_black = image.convert('L')
    
    image_array = np.array(img_black)
    
    # Reshape et Normalisation 
    image_final = image_array.reshape(1, 28*28).astype("float32") / 255 
    
    return image_final 



