from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "StudentScorePrediction.pkl")

model = joblib.load(MODEL_PATH)


@app.route("/")

def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get input from the form
    hours = float(request.form["hours"])

    # Validate input
    if hours < 0:
        return render_template(
        "index.html",
        error="Study hours cannot be negative!"
    )

    # Create DataFrame
    data = pd.DataFrame({"Hours": [hours]})

    # Predict
    prediction = model.predict(data)[0]

    # Return result
    return render_template(
        "index.html",
        prediction=round(prediction, 2),
        hours=hours
    )

if __name__ == "__main__":
    app.run(debug=True)