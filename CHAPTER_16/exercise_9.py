#!/usr/bin/env python3 

import time 
import calendar
import datetime
import numpy as np 
import tensorflow as tf 
import os 
import sys 
import pickle 

'''
    Train an Encoder-Decoder model that can convert a date string from one format to another(e.g. from "April 22, 2019" to "2019-04-22") MDY to ISO
'''
N_EMBED_DIM = 12

def dates_gen():

    initial_date = "January 1, 2027"
    end_date = "January 1, 2017"
    inital_dt = datetime.datetime.strptime(initial_date, "%B %d, %Y")
    end_dt = datetime.datetime.strptime(end_date, "%B %d, %Y")
    dates = [inital_dt - datetime.timedelta(days=days) for days in range(0, ( (inital_dt - end_dt) ).days + 1 )]
    dates = np.array(list(map(lambda element_dt: [element_dt.strftime("%B %d, %Y"), element_dt.strftime("%Y-%m-%d")], dates)))
    
    encoder_data = dates[:,0]
    decoder_data = dates[:,1]

    return encoder_data, decoder_data

def enc_preprocess(data):
    text_vectorization = tf.keras.layers.TextVectorization()
    ds = tf.data.Dataset.from_tensor_slices(data).map(lambda date_string: tf.strings.split(date_string)) 
    ds = ds.map(lambda arr: tf.concat( (arr, tf.constant(["<pad>","<pad>", "<pad>" ]))  , 0 ) )
    text_vectorization.adapt(ds)
    ds = ds.map(lambda x: tf.reshape(x, (-1,1)) )
    ds = ds.map(lambda x: x[::-1])
    return ds, text_vectorization

def dec_preprocess(data):
    text_vectorization = tf.keras.layers.TextVectorization(standardize=None)
    iso_split = lambda date_string:tf.strings.split(tf.strings.regex_replace(date_string, '[-]', ' - '   ))
    ds = tf.data.Dataset.from_tensor_slices(data).map(lambda date_string: tf.concat(( tf.constant(["<sos>"]), iso_split(date_string) , tf.constant(["<eos>"]) ), axis=0 ), num_parallel_calls=4 ) 
    ds = ds.map(lambda x: tf.reshape(x, (-1,1)) )
    text_vectorization.adapt(ds)
    return ds, text_vectorization

def process_prediction_iso(time_series):
    result = ''
    for [string] in time_series[::-1]:
        if '<' not in string:
            result += string 
    return result 

if sys.argv.__len__() <=1:
    assert(False)

if sys.argv[1] == 'train':

    # Preprocessing
    data_enc, data_dec = dates_gen()
    decoder_dataset, dec_text_vectorizer,= dec_preprocess(data_dec) 
    encoder_dataset, enc_text_vectorizer = enc_preprocess(data_enc)
    ds = tf.data.Dataset.zip((encoder_dataset,decoder_dataset))
    ds = ds.map(lambda e, d_full: ( (e, d_full[:-1]),(    tf.reshape(  dec_text_vectorizer(d_full[1:])  , (-1,1))  )) , num_parallel_calls=4)

    decoder_inputs = tf.keras.layers.Input(shape=[], dtype=tf.string)
    train_ds, test_ds = tf.keras.utils.split_dataset(ds, left_size=0.85)
    valid_ds, test_ds = tf.keras.utils.split_dataset(test_ds, left_size=0.5)

    # Encoder 
    encoder_inputs = tf.keras.layers.Input(shape=[], dtype=tf.string, name='enc_input')
    z = enc_text_vectorizer(encoder_inputs)
    z = tf.keras.layers.Embedding(len(enc_text_vectorizer.get_vocabulary())+ 2, N_EMBED_DIM, name='enc_embedding')(z)
    z = tf.keras.layers.LSTM(512, dropout=0.2, recurrent_dropout=0.2, return_sequences = True, name='enc_LSTM')(z)
    _, state_h, state_c  = tf.keras.layers.LSTM(512, dropout=0.2, recurrent_dropout=0.2, return_state=True, return_sequences = True, name='enc_LSTM2')(z)
    encoder_state = [ state_h, state_c ]

    # Decoder
    dec_vocab_size = len(dec_text_vectorizer.get_vocabulary())
    decoder_inputs = tf.keras.layers.Input(shape=[], dtype=tf.string, name='dec_input')
    z_dec = dec_text_vectorizer(decoder_inputs)
    z_dec = tf.keras.layers.Embedding(dec_vocab_size + 2, N_EMBED_DIM, name='dec_embedding')(z_dec)
    z_dec =  tf.keras.layers.LSTM(512,  dropout=0.2, recurrent_dropout=0.2, return_sequences = True, name='dec_LSTM') (sequences = z_dec, initial_state= encoder_state)
    z_dec =  tf.keras.layers.LSTM(512,  dropout=0.2, recurrent_dropout=0.2, return_sequences = True, name='dec_LSTM2') (z_dec)
    outputs = tf.keras.layers.TimeDistributed( tf.keras.layers.Dense(dec_vocab_size, activation='softmax') ) (z_dec)

    # Model
    model = tf.keras.Model(inputs=[encoder_inputs, decoder_inputs], outputs=[outputs])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    tf.keras.utils.plot_model(model, to_file=os.path.join(os.getcwd(), 'ex9_EncoderDecoder.png'), show_layer_names=True, show_layer_activations=True, show_trainable=True, show_shapes=True)
    
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint ("date_autoencoder.keras", save_best_only=True)
    history = model.fit( train_ds, epochs=7, validation_data=(valid_ds), callbacks=[checkpoint_cb])

    with open('date_autoencoder.pickel', 'wb') as file:
        pickle.dump({'decoder': {'vectorizer': dec_text_vectorizer} , 'encoder': {'vectorizer': enc_text_vectorizer}  }, file)

elif sys.argv[1] == 'infer':

    with open('date_autoencoder.pickel', 'rb') as file:
        pickle_data = pickle.load(file)
        v = pickle_data['decoder']['vectorizer']

        model = tf.keras.models.load_model(os.path.join(os.getcwd(), 'date_autoencoder.keras'))
        x_new_0 = np.array([['<pad>'], ['<pad>'], ['<pad>'], ['2022'],['02,'],['June']])
        
        current= [['<sos>']]
        n_samples = 6
        vocab = np.array(v.get_vocabulary())

        for i in range(n_samples):
            c = current + ([['']] * (n_samples  - 1 - i))
            c_new = tf.constant(np.array(c))
            pred = model.predict((  tf.constant(x_new_0), c_new))
            max_ = np.argmax(pred, -1)
            ind = np.reshape(max_,  (-1)) 
            pred_word = vocab[max_][i]
            current = current + [pred_word.tolist()]

        iso_string_pred = process_prediction_iso(current[1:])
        print(f'MDY = June 02, 2022\t Predicted ISO = {iso_string_pred}')


       