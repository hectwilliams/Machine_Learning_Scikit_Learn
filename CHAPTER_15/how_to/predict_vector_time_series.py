#!/usr/bin/env python3
'''
    Predict the next 10 samples using RNN 

'''
import os
import tensorflow as tf 
import generate_time_series
import numpy as np 
import matplotlib.pyplot as plt 
import sys 

n_steps = 50
n_next_steps = 10

if len(sys.argv) <=1:
    raise Exception('missing argument  ')

if sys.argv[1].lower() == 'predict':
    
    try:

        model = tf.keras.models.load_model(os.path.join(os.getcwd(), 'forecast2.keras'))
        
        series = generate_time_series(1, n_steps + n_next_steps)
        X_new, Y_new = series[:, :n_steps], series[:, n_steps:]
        Y_pred_n_next_steps = model.predict(X_new)[:,:, np.newaxis]

        x_time_samples = np.arange(0, 60)

        plt.subplot(1,1,1)
        plt.plot(x_time_samples, np.hstack((X_new, Y_new)).flatten() , label = 'actual')
        plt.scatter(x_time_samples, np.hstack((X_new, Y_new)).flatten() ,)
        plt.plot(x_time_samples, np.hstack((X_new, Y_pred_n_next_steps)).flatten() , label = 'predict')
        plt.scatter(x_time_samples, np.hstack((X_new, Y_pred_n_next_steps)).flatten() ,)
        plt.show()

    except ValueError as e:
        
        print('verify .keras file exist prior to executing model')

elif sys.argv[1].lower() == 'train':

    series = generate_time_series(20000, n_steps + n_next_steps)

    X_train, Y_train = series[:15000, :n_steps], series[:15000, -n_next_steps:, 0]
    X_valid, Y_valid = series[15000:18000, :n_steps], series[15000:18000, -n_next_steps:, 0]
    X_test, Y_test = series[18000:20000, :n_steps], series[18000:20000, -n_next_steps:, 0]

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(50,1,)),
        tf.keras.layers.SimpleRNN(20, return_sequences=True),
        tf.keras.layers.SimpleRNN(20, return_sequences=True),
        tf.keras.layers.SimpleRNN(20),
        tf.keras.layers.Dense(10)
    ])

    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam())
    history = model.fit(X_train, Y_train, epochs =5, validation_data=(X_valid, Y_valid))
    model.save(os.path.join(os.getcwd(), 'forecast2.keras'))