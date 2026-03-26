import numpy as np
from PIL import Image
import io

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image = image.resize((28, 28))
    array = np.array(image, dtype=np.float32) / 255.0
    array = array.reshape(1, 28, 28, 1)
    return array