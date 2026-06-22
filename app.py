from flask import Flask
from flask import render_template
from flask import request

import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(
    open("model.pkl", "rb")
)

@app.route("/")
def home():

    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    age = float(request.form["age"])

    gender = float(request.form["gender"])

    hours = float(request.form["hours"])

    attendance = float(
        request.form["attendance"]
    )

    data = np.array([
        [age,
         gender,
         hours,
         attendance]
    ])

    prediction = model.predict(data)

    return render_template(
        "index.html",
        result=round(prediction[0], 2)
    )

if __name__ == "__main__":
    app.run(debug=True)