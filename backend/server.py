import os
import warnings
import argparse
import yaml
import csv
import random
import string
import shutil
import json
import base64

from flask import Flask, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit, disconnect

app = Flask(__name__, static_folder="./build", static_url_path="/")
socketio = SocketIO(app, cors_allowed_origins="*", logger=True)


@app.route("/", defaults={"path": ""})
@app.route("/<path>")
def handler(path):
    return app.send_static_file("index.html")
