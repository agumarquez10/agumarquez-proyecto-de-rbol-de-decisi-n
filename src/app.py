from src.utils import db_connect
import pandas as pd
from src.utils import load_dataset

df = load_dataset()

# Cargar el dataset directamente
df = pd.read_csv("data/diabetes.csv")



# your code here
from flask import Flask, request, render_template
from pickle import load
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

app = Flask(__name__)
model = load(open("src/decision_tree_classifier_default_42.sav", "rb"))
class_dict = {
    "0": "Sin diabetes",
    "1": "Diabetes",
}
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

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Recoger los 8 valores del formulario en el orden correcto
        vals = [
            float(request.form["val1"]),  # Pregnancies
            float(request.form["val2"]),  # Glucose
            float(request.form["val3"]),  # BloodPressure
            float(request.form["val7"]),  # SkinThickness
            float(request.form["val8"]),  # Insulin
            float(request.form["val4"]),  # BMI
            float(request.form["val5"]),  # DiabetesPedigreeFunction
            float(request.form["val6"]),  # Age
        ]

        # Convertir a array y normalizar
        data = np.array([vals])
        data_normalized = scaler.transform(data)

        # Predicción con el modelo
        prediction = str(model.predict(data_normalized)[0])
        pred_class = class_dict[prediction]
    else:
        pred_class = None

    # Renderizar la plantilla con el resultado
    return render_template("index.html", prediction=pred_class)

