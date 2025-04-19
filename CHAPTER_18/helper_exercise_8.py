#! /usr/bin/env python3 

"""
    Uses tensorflow/serving docker container to make predictions on model

    Prerequisite:
    
        1) Model trained with script exercise_8.py

        2) Download docker and run the following commands:
            docker pull tensorflow/serving

            docker run -it --rm -p 8500:8500 -p 8501:8501 -v $ML_PATH/lunar_lander_model:/models/lunar_lander_model -e MODEL_NAME=lunar_lander_model tensorflow/serving

    Script run options:

        [1]  
            python helper_exercise_8.py rest 
        
        [2]  
            python helper_exercise_8.py rpc

"""
import json
import tensorflow as tf 
import requests 
from urllib.error import HTTPError
import os 
import numpy as np
from tensorflow_serving.apis.predict_pb2 import PredictRequest 
from tensorflow_serving.apis import prediction_service_pb2_grpc
import grpc
import sys 

x_new = tf.random.uniform((1, 8))
model_version = "0001"
model_name = "lunar_lander_model"

if len(sys.argv) < 2:
    raise ValueError('review script options above')

elif  sys.argv[1] == 'rest':
    input_data_json = json.dumps({
        "signature_name": "serving_default",
        "instances": x_new.numpy().tolist(),
    })

    domain = "www.yahoo.com"
    ret = os.system(f"ping -c 1 {domain}") # ret == 0 successful 
    print(ret)

    # scan for listening ports
    host = "localhost"
    port = "8501"
    ret = os.system(f"nc -vz {host} {port} ")
    print(ret) # expect pass 

    host = "localhost"
    port = "8503"
    ret = os.system(f"nc -vz {host} {port} ")
    print(ret) # expect fail 


    # get 
    url = 'http://localhost:8501/v1/models/lunar_lander_model/versions/0001'
    try:
        response = requests.get(url)
        response.raise_for_status()
        response_json = response.json() 
        print(response_json)
    except HTTPError as err:
        print(f'HTTP ERROR:\t{err.code}\t{err.reason}')

    # post (prediction)
    url = 'http://localhost:8501/v1/models/lunar_lander_model:predict'

    data_json = json.dumps({
        "signature_name": "serving_default",
        "instances":  [[0.36797094 ,0.71494186 ,0.43906498, 0.9908886 ,  0.8642545,  0.05222654 , 0.24658561, 0.28018057]],
    })

    try:
        response = requests.post(url, data=data_json)
        response.raise_for_status()
        response_json = response.json() 
        print(response_json)
    except HTTPError as err:
        print(f'HTTP ERROR:\t{err.code}\t{err.reason}')
        print(response.json())

elif sys.argv[1] == 'rpc':

    # create unsecure channel to server  
    channel = grpc.insecure_channel('localhost:8500') 

    # request specification
    grpc_request = PredictRequest() 
    grpc_request.model_spec.name = model_name 
    grpc_request.model_spec.signature_name = "serving_default"

    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel) # service (i.e stub) allow asynchronous calls to service Prediction Service

    # load amd convert input data 
    grpc_request.inputs['keras_tensor'].CopyFrom(tf.make_tensor_proto(x_new.numpy().tolist())) # convert input data to protocol buffer 

    # prediction
    predictions = stub.Predict(grpc_request, timeout=10.0) 
    outputs_proto = predictions.outputs['output_0']
    y_proba = tf.make_ndarray(outputs_proto)