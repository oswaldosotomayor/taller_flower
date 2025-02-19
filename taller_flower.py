# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 19:44:06 2025

@author: osotomayor
"""

import streamlit as st
import joblib
import numpy as np

# Cargar el modelo
modelo = joblib.load("modelo.pkl")

# Menú de navegación en el lado izquierdo (combo box)
st.sidebar.title("Menú de Navegación")
opcion = st.sidebar.selectbox(
    "Selecciona una opción:",
    ("Inicio", "Predicción", "Acerca de")
)

# Página de Inicio
if opcion == "Inicio":
    st.title("Bienvenido al Clasificador de Flores")
    st.write("""
    Esta aplicación utiliza un modelo de ML para predecir la especie de una Flor Iris basado en las características ingresadas.
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Iris_virginica.jpg/800px-Iris_virginica.jpg", caption="Flor Iris", use_container_width=True)

# Página de Predicción
elif opcion == "Predicción":
    st.title("Predicción de la Especie de Flor Iris")
    st.write("""
    Ingresa las características de la flor para predecir su especie.
    """)

    # Campos de entrada para las características
    sepal_length = st.number_input("Longitud del Sépalo (cm)", min_value=0.0, max_value=10.0, value=5.0)
    sepal_width = st.number_input("Ancho del Sépalo (cm)", min_value=0.0, max_value=10.0, value=3.0)
    petal_length = st.number_input("Longitud del Pétalo (cm)", min_value=0.0, max_value=10.0, value=1.0)
    petal_width = st.number_input("Ancho del Pétalo (cm)", min_value=0.0, max_value=10.0, value=0.2)

    if st.button("Predecir"):
        # Crear un array con las características ingresadas
        inputs = np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)

        # Realizar la predicción
        prediccion = modelo.predict(inputs)[0]

        # Mapear la predicción a un nombre de especie
        especies = {0: "Iris Setosa", 1: "Iris Versicolor", 2: "Iris Virginica"}
        especie_predicha = especies.get(prediccion, "Desconocida")

        # Mostrar la predicción
        st.success(f"La especie predicha es: **{especie_predicha}**")

        # Mostrar la imagen correspondiente a la especie predicha
        if especie_predicha == "Iris Setosa":
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg/800px-Kosaciec_szczecinkowaty_Iris_setosa.jpg", caption="Iris Setosa", use_container_width=True)
        elif especie_predicha == "Iris Versicolor":
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Iris_versicolor_3.jpg/800px-Iris_versicolor_3.jpg", caption="Iris Versicolor", use_container_width=True)
        elif especie_predicha == "Iris Virginica":
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Iris_virginica.jpg/800px-Iris_virginica.jpg", caption="Iris Virginica",use_container_width=True)

# Página Acerca de
elif opcion == "Acerca de":
    st.title("Acerca de")
    st.write("""
    Esta aplicación fue creada para predecir la especie de una flor Iris utilizando un modelo de Machine Learning.
    """)
    st.write("""
    **Tecnologías utilizadas:**
    - Streamlit para la interfaz de usuario.
    - Scikit-learn para el modelo de Machine Learning.
    """)
    st.write("""
    **Desarrollado por:**
    - Carolina Gamarra 
    - Jorge Caballero
    - Paul Morales
    - Oswaldo Sotomayor
    """)
    # Enlace a GitHub
    st.markdown("""
    **Referencia:**
    - [Visita mi perfil de GitHub](https://github.com/oswaldosotomayor/taller_flower)
    """)