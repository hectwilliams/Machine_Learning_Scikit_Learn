#!/usr/bin/env python3

'''



'''
import tensorflow as tf  
import os 
import numpy as np
import sys 
# from tensorflow.train import BytesList, FloatList, Int64List, Feature, Features, Example
import requests
import tarfile
import re

IMDB_DATASET_URL = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
WORKING_DIR = os.path.join(os.getcwd(), 'movie_review_dataset')
DEBUG_MODE = True
USER_REVIEW_MAX_LENGTH = 1000
N_TRAIN = 25000
BATCH_SIZE = 32 

def download_dataset(url, name):
    ''' 
        Download tar gz file 
    '''
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(os.path.join(WORKING_DIR, name) , 'wb') as f:
            f.write(response.raw.read())

def extract_gzip(file_path, extract_path):
    try:
        with tarfile.open(file_path, 'r:gz') as tar:
            tar.extractall(extract_path)
    except tarfile.TarError as e:
        print('tar.gz file extraction error')

def preprocess_pipeline( ds:tf.data.Dataset, string_tokenizer_func, embedding_func, batch_size, n_threads):
    ds = ds.map(lambda review, score: (string_tokenizer_func(review) , score), num_parallel_calls=n_threads)
    ds = ds.map(lambda tokens, score: ( tf.multiply( tf.reduce_mean(embedding_func(tokens), axis=0) , tf.sqrt( tf.cast(len(tokens), tf.float32)) ) , score), num_parallel_calls=n_threads)
    return ds.batch(batch_size).prefetch(1)

def preprocess_reviews(imdb_path, n_threads = 5):
    n_train = N_TRAIN
    shuffle_buffer = 250
    n_batch =  BATCH_SIZE
    text_vectorizer = tf.keras.layers.TextVectorization()

    train_path = os.path.join(imdb_path, 'train')
    test_path = os.path.join(imdb_path, 'test')

    train_pos_path = os.path.join(train_path,'pos')
    train_neg_path = os.path.join(train_path,'neg')
    test_pos_path = os.path.join(test_path,'pos')
    test_neg_path = os.path.join(test_path,'neg')
    
    train_pos_files = list(map(lambda filename: os.path.join(train_pos_path, filename) ,os.listdir(train_pos_path)))  
    train_neg_files = list(map(lambda filename: os.path.join(train_neg_path, filename) ,os.listdir(train_neg_path )))
    test_pos_files =  list(map(lambda filename: os.path.join(test_pos_path, filename)   ,os.listdir(test_pos_path) ))
    test_neg_files =  list(map(lambda filename: os.path.join(test_neg_path, filename)   ,os.listdir(test_neg_path) ))

    if DEBUG_MODE:
        n_batch = 1
        n_train = 2000
        n_train_over_2 = n_train // 2
        shuffle_buffer = 2000
        train_pos_files = train_pos_files[:n_train_over_2]
        train_neg_files = train_neg_files[:n_train_over_2]
        test_pos_files = test_pos_files[:n_train_over_2]
        test_neg_files = test_neg_files[:n_train_over_2]
    
    n_validation_set = int(n_train * 0.6)
    n_test_set = int(n_train * 0.4)

    dataset_train_pos = tf.data.TextLineDataset(train_pos_files)
    dataset_train_neg = tf.data.TextLineDataset(train_neg_files)
    dataset_train_full = dataset_train_pos.concatenate(dataset_train_neg)
    text_vectorizer.adapt(dataset_train_full)
    embedding_layer = tf.keras.layers.Embedding( output_dim = 2, input_dim=len(text_vectorizer.get_vocabulary()) + 2)

    dataset_train_pos_u = dataset_train_pos.map(lambda x: [x,0], num_parallel_calls=n_threads)
    dataset_train_neg_u = dataset_train_neg.map(lambda x: [x,1], num_parallel_calls=n_threads)
    dataset_train_full_u = dataset_train_pos_u.concatenate(dataset_train_neg_u).shuffle(shuffle_buffer)

    dataset_test_pos_u = tf.data.TextLineDataset(test_pos_files).map(lambda x: [x,0], num_parallel_calls=n_threads)
    dataset_test_neg_u = tf.data.TextLineDataset(test_neg_files).map(lambda x: [x,1], num_parallel_calls=n_threads)
    dataset_test_full_u = dataset_test_pos_u.concatenate(dataset_test_neg_u)
    
    dataset_validation_u = dataset_test_full_u.take(n_validation_set).shuffle(shuffle_buffer)
    dataset_test_u = dataset_test_full_u.skip(n_validation_set).shuffle(shuffle_buffer)
    
    dataset_train_full_u = preprocess_pipeline(dataset_train_full_u, text_vectorizer, embedding_layer, n_batch, n_threads) 
    dataset_validation_u = preprocess_pipeline(dataset_validation_u, text_vectorizer, embedding_layer, n_batch, n_threads) 
    dataset_test_u = preprocess_pipeline(dataset_test_u, text_vectorizer, embedding_layer, n_batch, n_threads) 

    return dataset_train_full_u, dataset_validation_u, dataset_test_u   

try :
    os.makedirs(WORKING_DIR)
except OSError as e:
    print(f'OSERROR:{e.errno} DIRECTORY EXISTS')

zipfile = re.search(r'.+\/(aclImdb_v1\.tar\.gz)', IMDB_DATASET_URL)
tar_filename = zipfile.groups(0)[0]

if not os.path.exists(os.path.join( WORKING_DIR, tar_filename)):
    download_dataset(IMDB_DATASET_URL, tar_filename)

if not os.path.exists(os.path.join(WORKING_DIR, 'aclImdb')):
    extract_gzip( os.path.join( WORKING_DIR, tar_filename), WORKING_DIR )

train_dataset, validation_dataset, test_dataset = preprocess_reviews(os.path.join(WORKING_DIR, 'aclImdb'))

# model Binary classifier ( Positive or Negative Movie)
input_= tf.keras.layers.Input(shape=[2])
hidden1 = tf.keras.layers.Dense(200, activation='relu')(input_)
hidden2 = tf.keras.layers.Dense(100, activation='relu')(hidden1)
hidden3 = tf.keras.layers.Dense(20, activation='relu')(hidden2)
hidden4 = tf.keras.layers.Dense(5, activation='relu')(hidden3)
output = tf.keras.layers.Dense(1, activation='sigmoid')(hidden4)
model = tf.keras.Model( inputs=[input_], outputs=[output])
print(model.summary())
model.compile( loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
history = model.fit(train_dataset, epochs=10, validation_data=validation_dataset)
