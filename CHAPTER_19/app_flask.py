#! /usr/bin/env python3
"""
    The code block runs in a docker tensor/flow container. The ENTRYPOINT was changed to run this Flask Server which will make predicts on the model at a certain port. 
"""

import json
import requests 
import os 
from flask import Flask, request, jsonify
from urllib.error import HTTPError
import subprocess 
import time 

DOCKER_MODEL_URL = "http://localhost:8501/v1/models/lunar_lander_model:predict"

app = Flask(__name__)

subprocess.Popen( args= ['bash' , '/usr/bin/tf_serving_entrypoint.sh' ] , stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL) # serving , redirect process i/o

@app.route('/predict', methods=['POST'])
def predict():

    try:
        response = requests.post(DOCKER_MODEL_URL, json=request.json)
        predictions = response.json()
        return jsonify(predictions)
    except HTTPError as err:
        print(f'HTTP ERROR:\t{err.code}\t{err.reason}')


"""
    Server' equivalent query = "curl -v -i -d \'{ \"instances\": [ <LIST> ]}\' -X POST http://localhost:8501/v1/models/lunar_lander_model:predict"
"""
