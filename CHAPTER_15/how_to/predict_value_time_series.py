#!/usr/bin/env python3
'''
    Predict the next 10 samples consecutively for a 50 sample wide wave signal having unformaly distrubted samples  

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
    raise Exception('missing argument')

if sys.argv[1].lower() == 'predict':

    try:
        model = tf.keras.models.load_model(os.path.join(os.getcwd(), 'forecast.keras'))
        series = generate_time_series(1, n_steps + n_next_steps)
        X_new, Y_new = series[:, :n_steps], series[:, -n_next_steps:]
        X = X_new 

        # predict at every step 

        for step_ahead  in range(n_next_steps):
            y_pred_next = model.predict(X[:, step_ahead:])[:, np.newaxis, :] 
            X = np.hstack((X, y_pred_next))

        y_predict_n_next_steps = X[:, -n_next_steps:] 
        x_axis_points = np.arange(0, n_steps + n_next_steps )

        plt.subplot(1, 1, 1)
        plt.plot( x_axis_points,  np.hstack((X_new, Y_new)).flatten(), label='data', c='blue')
        plt.scatter( x_axis_points, np.hstack((X_new, Y_new)).flatten(), c='purple', s=3)
        plt.plot( x_axis_points, np.hstack(X).flatten(), label='predict', c='purple')
        plt.scatter( x_axis_points, np.hstack(X).flatten(), c='purple', s=3)
        plt.legend()
        plt.show()

    except ValueError as e:
        
        print('verify .keras file exist prior to executing model')

elif sys.argv[1].lower() == 'train':
    series = generate_time_series(7000, n_steps= n_steps + 1)
    X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
    X_valid, y_valid = series[7000: 9000, :n_steps], series[7000: 9000, -1]
    X_test, y_test = series[9000:, :n_steps], series[9000:, -1]

    # Naive forecasting (MSE) 

    # y_pred_last_value = X_valid[:, -1] # last value of each validation wave 
    # mse_keras = tf.keras.losses.MeanSquaredError()
    # avg_loss = mse_keras(y_valid, y_pred_last_value) 

    # Naive forecasting (Linear Regression) 

    # model = tf.keras.Sequential([
    #         tf.keras.Input(shape=(50,)),
    #         tf.keras.layers.Flatten(),
    #         tf.keras.layers.Dense(1)
    #     ])
    # model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.SGD())
    # history = model.fit(X_train, y_train, epochs =20, validation_data=(X_valid, y_valid))

    # Single Recurrent Neuron

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(50,1,)),
        tf.keras.layers.SimpleRNN(20, return_sequences=True),
        tf.keras.layers.SimpleRNN(20, return_sequences=True),
        tf.keras.layers.SimpleRNN(1)
    ])
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam())
    history = model.fit(X_train, y_train, epochs =20, validation_data=(X_valid, y_valid))
    model.save(os.path.join(os.getcwd(), 'forecast.keras'))

