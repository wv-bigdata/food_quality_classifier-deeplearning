# Food quality classifier

---

### Creador:
- Wilbert Vong

---

**Este trabajo consiste en varios modelados de Deep Learning con CNN para clasificar el estado de alimentos. Además de la creación de una aplicación web para el uso de los modelos entrenados.**

**Rendimiento de los modelados**:

*Modelo para naranjas*: En Test obtuvo un Accuracy del 93.0%.

*Modelo para manzanas*: En Test obtuvo un Accuracy del 91.7%.

*Modelo para bananas*: En Test obtuvo un Accuracy del 99.8% *(claro overfitting)*.

---

El dataset de imágenes fue obtenido de [Kaggle](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification)

---

## Notebooks

[Modelado de CNN para naranjas, manzanas y bananas](https://colab.research.google.com/drive/1cRYTNzEyvCRkOg3h4_3FDwrx0aMN4on6?usp=sharing)

---

## Aplicación web con los modelos entrenados.

⚠️ La app web con problemas. Solo muestra la clasificación "Está fresca". ⚠️

🚧 Trabajando en solución o alternativa a Streamlit. 🚧

🔎 Los modelos funcionan correctamente en el entorno de Google Colab. 🔎

Luego de completar el modelado de CNN para los alimentos (naranjas, manzanas, bananas), he creado una aplicación web con Streamlit para poder hacer clasificaciones de una forma amigable. Para poder acceder a ella, haz click [aquí.](https://foodqualityclassifier-deeplearning-wv-bigdata.streamlit.app/)
