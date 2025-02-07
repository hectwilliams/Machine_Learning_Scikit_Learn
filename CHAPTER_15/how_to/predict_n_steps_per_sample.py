#!/usr/bin/env python3
'''
    Use RNN to predict the next 10 steps at each sample step

'''
import os
import tensorflow as tf 
from forecast import generate_time_series
import numpy as np 
import matplotlib.pyplot as plt 
import sys 
from matplotlib.lines import Line2D

@tf.keras.utils.register_keras_serializable()
def last_step_mse(y_expect_2D, y_predict_2D):
    mse = tf.keras.losses.MeanSquaredError()
    return mse(y_expect_2D[:, -1], y_predict_2D[:, -1])

n_steps = 50
n_next_steps = 10

if len(sys.argv) <=1:
    raise Exception('missing argument  ')

series = generate_time_series(20000, n_steps + n_next_steps)

X_train, Y_train = series[:15000, :n_steps], series[:15000, -n_next_steps:, 0]
X_valid, Y_valid = series[15000:18000, :n_steps], series[15000:18000, -n_next_steps:, 0]
X_test, Y_test = series[18000:20000, :n_steps], series[18000:20000, -n_next_steps:, 0]

if sys.argv[1].lower() == 'predict':
    
    try:

        model = tf.keras.models.load_model(os.path.join(os.getcwd(), 'forecast3.keras'))
        
        series = generate_time_series(1, n_steps + n_next_steps)
        X_new, Y_new = series[:, :n_steps], series[:, n_steps:]
        Y_pred_n_next_steps = model.predict(X_new)
        x_time_samples = np.arange(0, 60)

        plt.subplot(1,1,1)
        plt.plot(x_time_samples[:n_steps + 1], np.hstack((X_new, Y_new[:, :1])).flatten(), c='black')
        plt.scatter(x_time_samples[:n_steps], np.hstack((X_new)).flatten(), s=2, c='black')
        plt.plot(x_time_samples[-n_next_steps:], np.hstack((Y_new)).flatten(), c='black')
        plt.scatter(x_time_samples[-n_next_steps:], np.hstack((Y_new)).flatten(), c='black', s=2)

        for step_ahead in range(1, n_steps + 1):
            plt.plot(x_time_samples[ step_ahead - 1 : step_ahead + n_next_steps], np.hstack(( X_new[:,step_ahead-1 ], Y_pred_n_next_steps[:,step_ahead - 1])).flatten(), linestyle='dotted' )
            plt.scatter(x_time_samples[ step_ahead - 1 : step_ahead + n_next_steps], np.hstack(( X_new[:,step_ahead-1 ], Y_pred_n_next_steps[:,step_ahead - 1])).flatten() , s=1)

        legend_elements = [
            Line2D([0], [0], color='b', linestyle='dotted', label='pred_next_10_steps', c='black'),
            Line2D([0], [0], color='b', linestyle='solid', label='Actual steps', c='black'),
        ]

        plt.legend(handles=legend_elements)
        plt.show()

    except ValueError as e:
        
        print('verify .keras file exist prior to executing model')

elif sys.argv[1].lower() == 'train':

    # replace Y_train with timestep training vectors 

    Y = np.empty(shape=(20000, n_steps, 10)) # batch_id , sequence_in_length(i.e.row), n_steps(i.e. col)
    for step_ahead in range(1, 10 +1):
        sub_column = step_ahead - 1
        Y[:, :, sub_column] = series[:,  step_ahead: step_ahead + n_steps, 0]
    Y_train = Y[:15000]
    Y_valid = Y[15000:18000]
    Y_test = Y[18000:]

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(50,1,)),
        tf.keras.layers.SimpleRNN(20, return_sequences=True),
        tf.keras.layers.SimpleRNN(20, return_sequences=True),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(10)),
    ])

    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(), metrics=[last_step_mse])
    history = model.fit(X_train, Y_train, epochs =5, validation_data=(X_valid, Y_valid))
    model.save(os.path.join(os.getcwd(), 'forecast3.keras'))