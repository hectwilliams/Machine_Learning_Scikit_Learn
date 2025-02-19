#!/usr/bin/env python3 

'''
    Choose a particular embedded Reber grammer then train an RNN to identify whether a string respects that grammar or not.

'''

import numpy as np 
import tensorflow as tf 
import os 
import time 
import sys 

REBER_GRAPH = {
    # [next_pathname, next_node]  
    0: [ ['B', 1] ],
    1: [ ['T', 2] , ['P', 3] ]  ,
    2: [ ['S', 2] , ['X', 4] ]  ,
    3: [ ['T', 3] , ['V', 5] ]  ,
    4: [ ['X', 3] , ['S', 6] ]  ,
    5: [ ['P', 4] , ['V', 6] ]  ,
    6: [ ['E', -1]  ]  ,
    -1: []
}
DEBUG = False  
N_SAMPLES = 100 if DEBUG else 40000

def reber_gen(non_rebers=False, n_samples= 100) :
    result = ['' for _ in range(n_samples)]
    node_id = 0
    for i in range(n_samples):
        reber_string = ""
        node_id = 0
        while node_id != -1:
            routes = REBER_GRAPH[node_id]
            index = np.random.randint(len(routes))
            node_id = routes[index][1]
            reber_string += routes[index][0]
        # Add error to each string character. Equal probabiltiy of error to each string value rebel string 
        if non_rebers:
            reber_length = len(reber_string)
            for k in range(1, reber_length):
                update_char = np.random.randint(low= 0, high=2)
                if update_char:
                    # block of code ensures random selected is different
                    c = reber_string[k]
                    while  c == reber_string[k]:
                        c =REBER_GRAPH[np.random.randint(0,6)][0][0] 
                    reber_string = reber_string[:k] + c +reber_string[k+1:]
        result[i] = reber_string 
    return result

def preprocess(x_reber, x_nonreber):
    reber_ds = tf.data.Dataset.from_tensor_slices(x_reber).map(lambda token: (token, 1) )
    non_reber_ds = tf.data.Dataset.from_tensor_slices(x_nonreber).map(lambda token: (token, 0))
    dataset = non_reber_ds.concatenate(reber_ds).shuffle(N_SAMPLES).repeat(2).batch(1).prefetch(1)

    # Datasets 
    train = dataset.take( int(0.90 * N_SAMPLES))
    test = dataset.skip(int(0.90 * N_SAMPLES))
    valid = test.skip(int(0.10 * N_SAMPLES))
    test = test.take(int(0.10 * N_SAMPLES))

    return train , valid, test

def pred_threshold(model, x_new):
    x_new_tf = tf.data.Dataset.from_tensor_slices(x_new) 
    pred = model.predict( x_new_tf )
    return np.where(pred > 0.5, 1, 0)

if len(sys.argv) <=1:
    raise ValueError("Are we training or infering?")

if sys.argv[1] == 'train':

    # Generate reber data
    non_reber = reber_gen(True, N_SAMPLES)
    reber = reber_gen(False, N_SAMPLES)
    text_vectorization = tf.keras.layers.TextVectorization( output_sequence_length = 1)
    text_vectorization.adapt(non_reber + reber)
    num_vocab = text_vectorization.get_vocabulary()

    # Datasets 
    train_ds , valid_ds, test_ds = preprocess(reber, non_reber)

    # Model
    inputs = tf.keras.Input(shape=[], dtype=tf.string)
    z = text_vectorization(inputs)
    z = tf.keras.layers.Embedding(len(num_vocab) + 2, 6)(z)
    z = tf.keras.layers.GRU(2, return_sequences=False, dropout=0.1)(z)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(z)
    model= tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    print(model.summary())

    log_dir = os.path.join(  os.getcwd(), 'my_logs' ) 
    curr_log = os.path.join( log_dir, time.strftime("run_%Y_%m_%d-%H_%M_%S") )
    tensorboard_cb = tf.keras.callbacks.TensorBoard(curr_log)
    model.fit(train_ds, epochs=5, validation_data=valid_ds, callbacks=[tensorboard_cb])
    model.save(os.path.join(os.getcwd(), 'reber_classifcation.keras'))

elif sys.argv[1] == 'infer':

    model = tf.keras.models.load_model(os.path.join(os.getcwd(), 'reber_classifcation.keras'))
    reber_list = np.array(['BTSSXXTVVE', 'BPVVE', 'BTXXVPSE', 'BPVPXVPXVPXVVE', 'BTSXXVPSE' ])
    non_reber_list = np.array(['BTSSPXSE', 'BPTVVB', 'BTXXVVSE', 'BPVSPSE', 'BTSSSE'])

    x_new_true_values = np.array( reber_list)
    x_new = x_new_true_values[:,np.newaxis]
    y_pred = pred_threshold(model, x_new)

    print('Expect Truths', y_pred)

    x_new_true_values = np.array( non_reber_list)
    x_new = x_new_true_values[:,np.newaxis]
    y_pred = pred_threshold(model, x_new)

    print('Expect Falses', y_pred)
  
