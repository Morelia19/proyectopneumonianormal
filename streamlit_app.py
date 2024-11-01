import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import wget
import os
import contextlib

st.write(f"TensorFlow version: {tf.__version__}")

# Descargar y cargar el modelo completo si no existe
def download_and_load_model():
    model_url = 'https://dl.dropboxusercontent.com/s/4swm8f0ljha4m1ys743kb/best_model.keras?rlkey=spjl05988nt9jjvfujq2hrnc5&st=o3satd60'
    keras_model_path = 'best_model.keras'

    # Descargar el archivo si no existe
    if not os.path.exists(keras_model_path):
        try:
            st.write("Iniciando descarga del modelo...")
            wget.download(model_url, keras_model_path)
            st.success("Modelo descargado correctamente.")
        except Exception as e:
            st.error(f"Error al descargar el modelo: {e}")
            return None

    try:
        # Cargar el modelo completo en lugar de solo los pesos
        model = load_model(keras_model_path)
        st.success("Modelo cargado correctamente.")
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

    return model

# Cargar el modelo
model = download_and_load_model()

# Verificar si el modelo fue cargado correctamente
if model is None:
    st.error("No se pudo cargar el modelo.")
else:
    # Verificación de carga de archivo
    uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"], label_visibility="hidden")

    if uploaded_file is not None:
        # Mostrar la imagen subida
        st.image(uploaded_file, width=300, caption="Imagen cargada")

        # Preprocesamiento de la imagen para hacer la predicción
        img = image.load_img(uploaded_file, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Realizar la predicción con redirección de salida para evitar UnicodeEncodeError
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                prediction = model.predict(img_array)

        # Mostrar resultados
        if prediction[0][0] > 0.5:
            st.success('El modelo predice que la imagen es de un caso de **Pneumonia**.')
        else:
            st.success('El modelo predice que la imagen es de un caso **Normal**.')
