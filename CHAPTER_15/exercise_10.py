#!/usr/bin/env python3 

"""
    Download the Bach chorales dataset and unzip it. It is composed of 382 chorales composed by Johann Sebastian Bach. 
    Each chorale is 100 to 640 time steps long, and each time step contains 4 integers, where each integer corresponds to a note's index on a piano (Value of 0 means no note is played).
    Train a model--recurrent, convolutional, or both ...predict the next time step(4 notes) given a sequence of time steps.
"""

import os
import tensorflow as tf
import requests
import numpy as np 
import tarfile 
import csv
import sys 
import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D

URL_DATA = 'https://homl.info/bach'
N_NOTES = 4 # num of channels 
MAX_N_CHORALES_PER_CSV = 640 

def download_dataset(url=''):
    compress_filepath  = os.path.join(os.getcwd(), 'ds.tgz' )
    resp = requests.get(url, stream=True)
    if resp.status_code == 200:
        with open(compress_filepath, 'wb',  ) as f:
            f.write(resp.raw.read())
    with tarfile.open(compress_filepath, 'r:gz') as tar:
        tar.extractall(path = os.path.join(os.getcwd())) 

def preprocess(n_files):
    X = np.zeros( shape= (len(n_files), MAX_N_CHORALES_PER_CSV , N_NOTES ) , dtype=np.int32)
    Y = np.zeros( shape= (len(n_files), MAX_N_CHORALES_PER_CSV , N_NOTES ) , dtype=np.int32)
    
    for i in range(len(n_files)):
        filepath = n_files[i]
        with open(filepath, 'r') as csvfile:
            reader = csv.reader(csvfile)
            reader_n_lines_read = 0
            line = next(reader)  # skip header 
            try:
                while line:
                    line = next(reader)
                    X[i, reader_n_lines_read] = [int(value) for value in line]
                    reader_n_lines_read += 1
            except StopIteration as e:
                Y[i,: reader_n_lines_read-1] = X[i, 1: reader_n_lines_read]
    return X, Y

def get_datasets():
    chorales_dir = os.path.join(os.getcwd(), 'jsb_chorales' )
    train_dir = (os.path.join(chorales_dir, 'train'))
    test_dir = (os.path.join(chorales_dir, 'test'))
    valid_dir = (os.path.join(chorales_dir, 'valid'))

    train_list_dir = [os.path.join(train_dir, basename) for basename in  os.listdir(train_dir)]
    test_list_dir = [ os.path.join(test_dir, basename) for basename in  os.listdir(test_dir) ] 
    valid_list_dir = [ os.path.join(valid_dir, basename) for basename in  os.listdir(valid_dir) ] 
    Xtrain, Ytrain = preprocess(train_list_dir)
    Xtest, Ytest = preprocess(test_list_dir)
    Xvalid, Yvalid = preprocess(valid_list_dir)

    return Xtrain, Ytrain, Xtest, Ytest, Xvalid, Yvalid
    
def get_model():
    model= tf.keras.Sequential([
        tf.keras.Input(shape=(None,4)),
        tf.keras.layers.Conv1D(filters=20, kernel_size=10, strides=1, padding='same'), 
        tf.keras.layers.Conv1D(filters=20, kernel_size=15, strides=1, padding='same'), 
        tf.keras.layers.Conv1D(filters=20, kernel_size=5, strides=1, padding='same'), 
        tf.keras.layers.GRU(40, return_sequences=True),
        tf.keras.layers.GRU(40, return_sequences=True),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(4, activation='softmax')),
    ])
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

def simple_model():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(None, 4)),
        tf.keras.layers.Conv1D(filters=30, kernel_size=10, strides=1, padding='same'),
        tf.keras.layers.Conv1D(filters=30, kernel_size=5, strides=1, padding='same'),
        tf.keras.layers.Conv1D(filters=30, kernel_size=2, strides=1, padding='same'),
        tf.keras.layers.GRU(40, return_sequences=True),
        tf.keras.layers.GRU(40, return_sequences=True),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(4)),
    ])
    model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=0.03, momentum=0.4), metrics=['accuracy'])

    return model 

def show_estimated_notes(next_vectors, y):
    n_vectors= len(next_vectors)
    col_size = N_NOTES*n_vectors - 1
    buffer = np.zeros(shape= (n_vectors*2, col_size))
    steps = np.arange(col_size)
    
    fig = plt.figure()
    fig.suptitle('Predict Next 4 Chorales Notes')

    for i in range(2):
        w_true = np.arange(i, 4 + i)
        w_next = np.arange(i + 1, 4 + i + 1)
        buffer[i*2 ,w_true ] = next_vectors[i]
        buffer[i*2 + 1, w_next] = y[i]
        
        plt.subplot(1,2,i + 1 )
        plt.plot(steps[w_true], buffer[i*2 , w_true], c='black')
        plt.scatter(steps[w_true], buffer[i*2 , w_true], marker='x', c='black')
        
        plt.plot(steps[ w_next], buffer[i*2 + 1, w_next], c='orange', linestyle='dotted')
        plt.scatter(steps[ w_next], buffer[i*2 + 1, w_next], marker='x', c='orange')
    
    fig.legend([Line2D([0], [0], color='black'), Line2D([0], [0], color='orange',  linestyle='dotted')], ['True notes', 'generative-next notes', ])
    plt.show() 


if len(sys.argv) <2:
    raise ValueError(f'missing scriptt task. Selections are \'train\' or \'predict\' ')

download_dataset(URL_DATA)
X_train, y_train, X_valid, y_valid, X_test, y_test =  get_datasets()

if sys.argv[1] =='train':
    model = simple_model()
    history = model.fit(X_train, y_train, epochs=10)
    model.save(os.path.join(os.getcwd(), 'chorales.keras'))

elif sys.argv[1] == 'predict':
    model = tf.keras.models.load_model(os.path.join(os.getcwd(), 'chorales.keras'))
    X_new = X_test[:1, 3:5] # first instance , time steps 3,and 4
    y_pred = model.predict(X_new)
    print(X_new.shape)
    print(y_pred.shape)
    show_estimated_notes(X_new[0], y_pred[0])

