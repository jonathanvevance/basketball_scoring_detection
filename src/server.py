from os import close
from typing import Dict
from flask import Flask, request, jsonify
from flask_cors import CORS
import csv
import time
import predict

app = Flask(__name__, static_folder="./build", static_url_path="/")


@app.route("/", defaults={"path": ""})
@app.route("/<path>")
def handler(path):
    return app.send_static_file("index.html")


@app.route("/fileupload", methods=["GET", "POST"])
def index():
    # result = request.form['word'] in dictionary
    result = {"result": "success"}
    return jsonify(result)


@app.route("/getvalue", methods=["GET", "POST"])
def getProbability():
    is_scoring = predict.predict()
    # result = request.form['word'] in dictionary
    lineData = []
    with open("/home/amark/Projects/MTX-HackOlympics/reports/probability_values.csv") as file:
        reader = csv.DictReader(file, delimiter=",")
        for index, row in enumerate(reader):
            lineData.append({"time": row["time"], "value": row["values"]})
    # time.sleep(10)
    print("Sending")
    return jsonify(lineData)


@app.route("/getvideodata", methods=["GET", "POST"])
def getVideoProbab():
    lineData = []
    with open("/home/amark/Projects/MTX-HackOlympics/reports/probability_values.csv") as file:
        reader = csv.DictReader(file, delimiter=",")
        for index, row in enumerate(reader):
            lineData.append({"time": row["time"], "value": row["values"], "fps": row["fps"]})
    # time.sleep(10)
    print("Sending")
    return jsonify(lineData)


if __name__ == "__main__":
    # app.config['SECRET_KEY'] = 'BlaBlaBla'
    CORS(app)
    app.run(port=4000, debug=True)
