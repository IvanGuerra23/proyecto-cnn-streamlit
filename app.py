import json

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Clasificador de Flores", page_icon="🌸", layout="centered")

MODEL_PATH = "models/flowers_cnn.keras"
CLASS_NAMES_PATH = "models/class_names.json"
IMG_SIZE = (160, 160)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMG_SIZE)
    array = tf.keras.utils.img_to_array(image)
    return np.expand_dims(array, axis=0)


def main():
    st.title("🌸 Clasificador de Flores con CNN")
    st.write(
        "Sube una imagen de una flor (JPG o PNG) y el modelo predecirá "
        "a cuál de las 5 clases pertenece."
    )

    model = load_model()
    class_names = load_class_names()

    uploaded_file = st.file_uploader(
        "Selecciona una imagen", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen subida", use_container_width=True)

        input_array = preprocess(image)
        predictions = model.predict(input_array, verbose=0)[0]

        top_index = int(np.argmax(predictions))
        top_class = class_names[top_index]
        top_prob = float(predictions[top_index])

        st.success(f"**Predicción: {top_class}** — Probabilidad: {top_prob * 100:.2f} %")

        st.subheader("Probabilidades por clase")
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(class_names, predictions * 100, color="steelblue")
        bars[top_index].set_color("seagreen")
        ax.set_ylabel("Probabilidad (%)")
        ax.set_ylim(0, 100)
        ax.bar_label(bars, fmt="%.1f%%")
        plt.xticks(rotation=30, ha="right")
        st.pyplot(fig)


if __name__ == "__main__":
    main()
