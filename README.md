# Food quality classifier

---

### Creador:
- Wilbert Vong

---

**Este trabajo consiste en varios modelados de Deep Learning con CNN para clasificar el estado de alimentos. Además de la creación de una aplicación web para el uso de los modelos entrenados.**

**Rendimiento de los modelados**:

*Modelo para naranjas*: En Test obtuvo un Accuracy del 94.8%.

*Modelo para manzanas*: En Test obtuvo un Accuracy del 98.7% *(claro overfitting)*.

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

🍎 [Aplicando el modelo de manzanas en Google Colab](https://colab.research.google.com/drive/1hud2xEeDdU1-qAv48q7Nj0Vv2jWhaan_?usp=sharing)

🍊 [Aplicando el modelo de naranjas en Google Colab](https://colab.research.google.com/drive/1AeKtykonhfhhT2g2ctTDi04eRi2oyXjN?usp=sharing)

🍌 [Aplicando el modelo de naranjas en Google Colab](https://colab.research.google.com/drive/1PFwip__JW0yYBQUGnvmDyP2SIUfo9xSi?usp=sharing)

---

~~Luego de completar el modelado de CNN para los alimentos (naranjas, manzanas, bananas), he estado buscando una manera amigable para usar los modelos por medio de una app web. Se intentó con Streamlit, pero el modelo solo dice que las imágenes representan frutas frescas, haz click [aquí.](https://foodqualityclassifier-deeplearning-wv-bigdata.streamlit.app/)~~
