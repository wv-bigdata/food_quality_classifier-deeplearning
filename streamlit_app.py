import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Función para cargar y preprocesar la imagen
def preprocess_image(image):
    image = image.resize((150, 150))
    image = np.array(image) / 255.0  # Normalizar píxeles
    image = np.expand_dims(image, axis=0)
    return image

# Cargar modelos
apple_model = tf.keras.models.load_model('apple_classifier_model.h5')
orange_model = tf.keras.models.load_model('orange_classifier_model.h5')
banana_model = tf.keras.models.load_model('banana_classifier_model.h5')

# Clases de los alimentos
FOOD_CLASSES = {
    'Manzana': ['Fresca', 'Dañada'],
    'Naranja': ['Fresca', 'Dañada'],
    'Banana': ['Fresca', 'Dañada']
}

# Encabezado de la aplicación
st.title('Clasificador de estados de Alimentos')

# Selección del alimento
food = st.selectbox('Selecciona un alimento:', ['Manzana', 'Naranja', 'Banana'])

# Cargar imagen
uploaded_image = st.file_uploader('Carga una imagen del alimento:', type=['jpg', 'jpeg', 'gif', 'png'])

# Mostrar imagen si se carga
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Imagen cargada', use_column_width=True)

# Botón para clasificar siempre presente
if st.button('Clasificar'):
    if uploaded_image is not None:
        # Preprocesar imagen
        preprocessed_image = preprocess_image(image)

        # Seleccionar modelo y hacer predicción
        if food == 'Manzana':
            prediction = apple_model.predict(preprocessed_image)
            prediction_label = FOOD_CLASSES[food][int(prediction[0, 0])]
        elif food == 'Naranja':
            prediction = orange_model.predict(preprocessed_image)
            prediction_label = FOOD_CLASSES[food][int(prediction[0, 0])]
        elif food == 'Banana':
            prediction = banana_model.predict(preprocessed_image)
            prediction_label = FOOD_CLASSES[food][int(prediction[0, 0])]

        # Mostrar resultado de predicción
        st.write(f'Resultado de clasificación: Está {prediction_label}.')
    else:
        st.write('Por favor, carga una imagen del alimento para realizar la clasificación.')

# Pie de página
st.caption("Clasificador de estados de alimentos desarrollado por Wilbert Vong - Big Data Architect.")
