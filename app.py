from flask import Flask, request, render_template
import joblib
import numpy as np


app = Flask(__name__)

model = joblib.load("decision_tree_model.pkl")


@app.route("/")
def home():
    return render_template("form.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get input values from the HTML form
    hours = float(request.form["hours"])         # Now user enters 2, 5, 10
    attendance = float(request.form["attendance"])
    tutoring = int(request.form["tutoring"])         # 0 or 1
    region = int(request.form["region"])             # 0, 1, or 2
    parent = int(request.form["parent"])             # 0, 1, or 2

    # Feature Engineering (same as training)
    study_fam = hours * parent
    study_att = hours * attendance

    # Combine features into one array
    input_features = np.array([[hours, attendance, tutoring, region, parent, study_fam, study_att]])

    # Make prediction
    prediction = model.predict(input_features)[0]

    return render_template("form.html", prediction=round(prediction, 2))

if __name__ == "__main__":
    app.run(debug=True)