import argparse
import json

import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = "models/flowers_cnn.keras"
CLASS_NAMES_PATH = "models/class_names.json"
IMG_SIZE = (160, 160)


def load_image(path: str) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize(IMG_SIZE)
    array = tf.keras.utils.img_to_array(image)
    return np.expand_dims(array, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Predice la clase de una imagen con el modelo CNN de flores."
    )
    parser.add_argument("image", help="Ruta de la imagen (jpg/png)")
    args = parser.parse_args()

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    input_array = load_image(args.image)
    predictions = model.predict(input_array, verbose=0)[0]

    top_index = int(np.argmax(predictions))
    print(f"Clase: {class_names[top_index]}")
    print(f"Probabilidad: {predictions[top_index] * 100:.2f} %")


if __name__ == "__main__":
    main()
