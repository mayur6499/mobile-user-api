import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("mobile_user_model.pkl")

@app.route("/", methods=["GET", "POST"])

def index():
    prediction = None

    if request.method == "POST":
        app_usage = float(request.form["app_usage"])
        battery = float(request.form["battery"])
        apps = int(request.form["apps"])
        data = float(request.form["data"])
        screen = float(request.form["screen"])
        age = int(request.form["age"])
        gender = int(request.form["gender"])
        os = int(request.form["os"])
        location = int(request.form["location"])

        features = np.array([[app_usage, battery, apps, data, screen, age,gender, os, location]])
        prediction = model.predict(features)[0]

    return render_template("index.html", prediction=prediction)
@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not request.is_json:
        return jsonify({"error": "Invalid JSON"}), 400
    data = request.json

    app_usage = float(data["app_usage"])
    battery = float(data["battery"])
    apps = int(data["apps"])
    data_usage = float(data["data"])
    screen = float(data["screen"])
    age = int(data["age"])
    gender = int(data["gender"])
    os = int(data["os"])
    location = int(data["location"])

    features = np.array([[app_usage, battery, apps, data_usage, screen, age, gender, os, location]])
    prediction = int(model.predict(features)[0])

    return jsonify({
        "prediction": prediction
    })
@app.route("/api/predict", methods=["GET"])
def api_predict1():
    app_usage = float(request.args.get("app_usage"))
    battery = float(request.args.get("battery"))
    apps = int(request.args.get("apps"))
    data = float(request.args.get("data"))
    screen = float(request.args.get("screen"))
    age = int(request.args.get("age"))
    gender = int(request.args.get("gender"))
    os = int(request.args.get("os"))
    location = int(request.args.get("location"))

    features = np.array([[app_usage, battery, apps, data, screen, age, gender, os, location]])
    prediction = model.predict(features)[0]

    return jsonify({"prediction": float(prediction)})

if __name__ == "__main__":
    app.run(debug=True)
