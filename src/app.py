import streamlit as st
import pandas as pd
import numpy as np
from pickle import load
from sklearn.preprocessing import StandardScaler

# Cargar modelo
model = load(open("src/decision_tree_classifier_default_42.sav", "rb"))

# Diccionario de clases
class_dict = {
    "0": "Sin diabetes",
    "1": "Diabetes",
}

# Dataset para ajustar el scaler
df = pd.read_csv("data/diabetes.csv")

num_variables = [
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age'
]

scaler = StandardScaler()
scaler.fit(df[num_variables])

# --- Interfaz Streamlit ---
st.title("Predicción de Diabetes con Árbol de Decisión")

# Inputs del usuario
pregnancies = st.number_input("Número de embarazos", min_value=0, max_value=20, value=1)
glucose = st.number_input("Nivel de glucosa", min_value=0, max_value=300, value=120)
blood_pressure = st.number_input("Presión sanguínea", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Espesor de piel", min_value=0, max_value=100, value=20)
insulin = st.number_input("Nivel de insulina", min_value=0, max_value=900, value=80)
bmi = st.number_input("Índice de masa corporal (BMI)", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Función de pedigrí de diabetes", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Edad", min_value=0, max_value=120, value=30)

# Botón de predicción
if st.button("Predecir"):
    vals = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
    data = np.array([vals])
    data_normalized = scaler.transform(data)

    prediction = str(model.predict(data_normalized)[0])
    pred_class = class_dict[prediction]

    st.success(f"Resultado: {pred_class}")
