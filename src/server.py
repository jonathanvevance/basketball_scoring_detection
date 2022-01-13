"""Function to start the python server. Loads the flask app and runs it. Provides the frontend with data on fetch
    

Returns:
    none: Returns nothing.
"""

from os import close
from flask import Flask, request, jsonify
from flask_cors import CORS
import csv
import time
import predict
import shutil

app = Flask(__name__, static_folder="./build", static_url_path="/")


@app.route("/", defaults={"path": ""})
@app.route("/<path>")
def handler(path):
    """[Return the index.html file from frontend build]
    Args:
        path (any): The data provided by the user when called from browser or from the server
    Returns:
        file: [Return the index.html file from frontend build]
    """
    return app.send_static_file("index.html")


@app.route("/fileupload", methods=["GET", "POST"])
def index():
    """Returns success/failure of file upload to NodeJS server.

    Returns:
        JSON: JSON String
    """
    result = {"result": "success"}
    return jsonify(result)


@app.route("/getvalue", methods=["GET", "POST"])
def getProbability():
    """Returns probability of scoring when called from frontend. Calls predict() and runs inference. Writes the data to csv file.

    Returns:
        JSON: JSON String of frame probabilities
    """

    shutil.copy(
        "data/inference/video_upload/video.mp4", "src/server/assets/video.mp4"
    )  # copy video from inference folder to assets folder
    is_scoring = predict.predict()
    lineData = []
    with open("reports/probability_values.csv") as file:  # open csv file
        reader = csv.DictReader(file, delimiter=",")
        for index, row in enumerate(reader):
            lineData.append(
                {"time": row["time"], "value": row["values"], "is_scoring": row["is_scoring"]}
            )  # append data to list
    print("Sending")
    return jsonify(lineData)  # return list of data


@app.route("/getvideodata", methods=["GET", "POST"])
def getVideoProbab():
    """Called when video is loaded in frontend. Returns the video frame probability data.

    Returns:
        JSON: JSON String of frame probabilities and is_scoring and fps
    """

    lineData = []
    with open("reports/probability_values.csv") as file:  # open csv file
        reader = csv.DictReader(file, delimiter=",")
        for index, row in enumerate(reader):
            lineData.append(
                {
                    "time": row["time"],
                    "value": row["values"],
                    "fps": row["fps"],
                    "is_scoring": row["is_scoring"],
                }  # append data to list
            )
    print("Sending")
    return jsonify(lineData)  # return list of data


if __name__ == "__main__":
    """Runs the server at port 4000"""
    CORS(app)
    app.run(host='0.0.0.0',port=4000, debug=True)
