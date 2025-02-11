#!/usr/bin/env python3
"""
    Train a classification model using SketchRNN dataset. 

    The model predicts pen status while drawing strokes: pen is writing or not writing. 

    Many character based language frameworks have certain stroke patterns. Being able to classify stroke pattens may provide insight into whether stroke patterns represent Kanji or not. 

    The model could be used to create 'distorted' Kanji as the predictions always have some inherit error. 
"""
import os 
import sys
import numpy as np
import requests
import matplotlib.pyplot as plt 
import tensorflow as tf

SKETCH_RNN_DATASET = "https://github.com/hardmaru/sketch-rnn-datasets/raw/refs/heads/master/kanji/short_kanji.npz"
MAX_SEQ_SIZE = 88 
TEST_IMAGE = np.array([[0,-5,0],[3,-23,0],[8,-14,0],[27,-36,0],[22,-19,0],[53,-29,0],[21,-5,0],[30,-1,0],[25,6,0],[28,12,0],[33,29,0],[33,54,0],[16,38,0],[-85,0,0],[-117,-14,0],[-49,0,0],[-56,10,1],[60,-80,0],[23,-4,0],[84,0,0],[45,-9,0],[20,0,1],[-177,42,0],[182,5,0],[39,6,1],[-160,-77,0]])
USE_WAVENET_MODEL = False 

@tf.keras.utils.register_keras_serializable()
def last_step_mse(y_expect_2D, y_predict_2D):
    mse = tf.keras.losses.MeanSquaredError()
    return mse(y_expect_2D[:, -1], y_predict_2D[:, -1])

def preprocess_input_padding(list_of_vlists):
    length = len(list_of_vlists)
    y = np.zeros(shape=(length, MAX_SEQ_SIZE, 1),dtype=np.int16)
    x = np.zeros(shape=(length, MAX_SEQ_SIZE, 2 ), dtype=np.int16)

    for i in range(length):
        vlist = list_of_vlists[i]
        n_zero_record = MAX_SEQ_SIZE - len(vlist)
        if n_zero_record:
            null_vectors = np.array([[0,0]] * n_zero_record)
            x_ = vlist[:, :2]
            y_ = vlist[:,2]
            x[i,:] = np.vstack((x_, null_vectors ))
            y[i,:] = np.hstack((y_, (null_vectors[:,0] + 2) ))[:, np.newaxis]
    return x, y

def draw_stroke(example, plt_api):
    x_curr, y_curr  = 0, 0
    pen_prev_lifted = True
    liftsize= 5

    for x, y, pen_lifted in example:
        x_next, y_next = x_curr + x, y_curr - y

        if not pen_lifted:

            if pen_prev_lifted:
                plt_api.scatter( [x_next], [y_next], c= 'darkorange', s= liftsize)
            else:
                plt_api.plot([x_curr, x_next], [y_curr, y_next], c= 'black')

        else:

            if not pen_prev_lifted:
                # if pen was previously lifted an error occured in the data acquisition tool or model
                plt_api.plot([x_curr, x_next], [y_curr, y_next], c= 'black')
                plt_api.scatter([x_next], [y_next], c='red', s=liftsize )

        pen_prev_lifted = pen_lifted
        x_curr, y_curr  = x_next, y_next

def show_pen_pred(stroke_data, pen_events, model):
    n_tests_samples = 10
    
    fig = plt.figure()
    fig.tight_layout()

    for i in range(0, n_tests_samples,2):
        pen_status = pen_events[i]
        pen_stroke_data = stroke_data[i ]
        pred = model.predict(stroke_data[i][np.newaxis, :])
        sparse_cross_entropy= tf.keras.losses.SparseCategoricalCrossentropy()
        error = np.round(sparse_cross_entropy( pen_status[np.newaxis, :], pred), 2).__str__()

        for k in range(1,3 ):
            ax = plt.subplot(n_tests_samples, 2, i + k )
            plt.axis('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                plt.title('Actual Kanji Pen Strokes', fontsize = 10) if k ==1 else plt.title('Predicted Kanji Pen Strokes', fontsize = 10  )
            draw_stroke(np.c_[(pen_stroke_data, pen_status)], plt)
            if k %2 == 0:
                ax.annotate(f'MSE = {error}', xy=(1, 1), xycoords='axes fraction', xytext=(0, 0), textcoords='offset points', bbox=dict(boxstyle="round", fc="1"), fontsize = 9)
                
def get_dataset(remote_npz_path=''):
    local_npz_path = os.path.join(os.getcwd(), 'data.npz')

    if len(sys.argv) <2:
        raise ValueError
    
    if sys.argv[1] == 'predict':

        try:
            data = np.load(local_npz_path, mmap_mode='r', allow_pickle=True, encoding='bytes')
        except:
            raise Exception()
        
        test_set = data['test']
        X_test , y_test = preprocess_input_padding(test_set)
        if USE_WAVENET_MODEL:
            model = tf.keras.models.load_model(os.path.join(os.getcwd(), 'sketch_wavenet_model.keras'))
        else:
            model = tf.keras.models.load_model(os.path.join(os.getcwd(), 'sketch_basic_model.keras'))
        show_pen_pred(X_test, y_test,  model )

        plt.show()
    elif sys.argv[1] == 'train':

        try:
            response = requests.get(remote_npz_path, stream=True)
            if response.status_code == 200:
                with open( local_npz_path, 'wb') as f:
                    f.write(response.raw.read())
        except:
            raise Exception()
        
        data = np.load(local_npz_path, mmap_mode='r', allow_pickle=True, encoding='bytes')
        train_set = data['train']
        X_train , y_train = preprocess_input_padding(train_set)

        if USE_WAVENET_MODEL:
            wavenet_model = tf.keras.Sequential()
            wavenet_model.add(tf.keras.Input(shape=(None,2)))
            for rate in (1,2,3,4,8,16) * 2 :
                wavenet_model.add(tf.keras.layers.Conv1D(filters=20, kernel_size=2, padding='causal', dilation_rate = rate, activation='softplus'))
            wavenet_model.add(tf.keras.layers.Conv1D(filters=3, kernel_size = 1, activation='softmax') )
            wavenet_model.compile(loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.SGD(learning_rate=0.08) , metrics = ['accuracy'])
            wavenet_model.fit(X_train, y_train, epochs=3)
            wavenet_model.save(os.path.join(os.getcwd(), 'sketch_wavenet_model.keras'))
        else: 
            basic_model = tf.keras.Sequential([
                tf.keras.Input(shape=(None,2)),
                tf.keras.layers.Conv1D(filters=20, kernel_size=10, strides=1, padding='same'), 
                tf.keras.layers.Conv1D(filters=20, kernel_size=15, strides=1, padding='same'), 
                tf.keras.layers.Conv1D(filters=20, kernel_size=5, strides=1, padding='same'), 
                tf.keras.layers.GRU(40, return_sequences=True),
                tf.keras.layers.GRU(40, return_sequences=True),
                tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(3, activation='softmax')),
            ])
            basic_model.compile(loss="sparse_categorical_crossentropy", optimizer='sgd', metrics = ['accuracy'])
            basic_model.fit(X_train, y_train, epochs=3)
            basic_model.save(os.path.join(os.getcwd(), 'sketch_basic_model.keras'))
    
if __name__ == '__main__':
    get_dataset(SKETCH_RNN_DATASET)

