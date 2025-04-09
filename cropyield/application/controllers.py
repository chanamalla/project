from flask import render_template, request
from flask import current_app as app
from model.model_prediction import predict
import numpy as np
import joblib

@app.route("/", methods = ["GET", "POST"])
def home():
    op = -1
    if request.method == "POST":
        rainfall = float(request.form["rainfall"]) if request.form["rainfall"] else 0
        pesticide = float(request.form["pesticide"]) if request.form["pesticide"] else 0
        temperature = float(request.form["temperature"]) if request.form["temperature"] else 0
        op = predict(rainfall,pesticide ,temperature)

    return render_template("index.html", prediction = op)