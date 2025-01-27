#!/usr/bin/env python3

'''
    Download a dataset, split it, create a tf.data.Dataset to load it and preprocess it efficiently, then build 
    and train a binary classification model containing an Embedding Layer
'''

import tensorflow as tf  
import os 
import requests
import tarfile
import re
import tensorflow_datasets as tfds 
import sys 

IMDB_DATASET_URL = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
WORKING_DIR = os.path.join(os.getcwd(), 'movie_review_dataset')
N_TRAIN = 25000
BATCH_SIZE = 32
EMBEDDING_DIM = 30
USE_TFDS = True
DEBUG_MODE = True

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

def preprocess_pipeline( ds:tf.data.Dataset, string_tokenizer_func, embedding_func, batch_size, n_threads=5):
    '''
        Preprocess: TextVectorization ->Embedding
    '''
    ds = ds.map(lambda review, score: (string_tokenizer_func(review) , score), num_parallel_calls=n_threads)
    ds = ds.map(lambda tokens, score: ( tf.multiply( tf.reduce_mean(embedding_func(tokens), axis=0) , tf.sqrt( tf.cast(len(tokens), tf.float32)) ) , score), num_parallel_calls=n_threads)
    return ds.batch(batch_size, drop_remainder=True, num_parallel_calls=n_threads).prefetch(1)

def get_datasets(imdb_path, use_tfds, debug = False):
    '''
        Downloads reviews from webpage or downloads using TFDS; followed by preprocessing pipelines.
    '''
    text_vectorizer = tf.keras.layers.TextVectorization
    embedding_layer = tf.keras.layers.Embedding
    n_train = n_test = n_validation = shuffle_buffer = N_TRAIN
    n_batch = BATCH_SIZE
    n_threads = 5

    if debug:
        n_batch = 5
        n_train = shuffle_buffer = 2000
    
    n_train_over_2 = n_train // 2
    n_validation = int(n_test * 0.6)
    n_test  = int(n_test * 0.4)
    text_vectorizer = text_vectorizer()

    if use_tfds:
        ds = tfds.load(name="imdb_reviews")
        imdb_train_ds = ds['train'].take(n_train)
        imdb_test_ds = ds['test'].take(n_train)
        
        reviews_ds = imdb_train_ds.map(  lambda dict_: dict_['text'] , num_parallel_calls=n_threads)
        train_ds = imdb_train_ds.map(lambda dict_: (dict_['text'],dict_['label'] ) , num_parallel_calls=n_threads) 
        test_ds  = imdb_test_ds.map(lambda dict_: (dict_['text'],dict_['label'] ) , num_parallel_calls=n_threads)
        
        text_vectorizer.adapt( reviews_ds)
        embedding_layer = embedding_layer( output_dim = EMBEDDING_DIM, input_dim=len(text_vectorizer.get_vocabulary()))
        
        validation_ds = test_ds.take(n_validation)
        test_ds = test_ds.skip(n_validation)
        
        train_ds = train_ds.cache()
        test_ds = test_ds.cache()
        validation_ds = validation_ds.cache()

        train_ds = train_ds.shuffle(shuffle_buffer).repeat(1)

    else:

        train_path = os.path.join(imdb_path, 'train')
        test_path = os.path.join(imdb_path, 'test')

        train_pos_path = os.path.join(train_path,'pos')
        train_neg_path = os.path.join(train_path,'neg')
        test_pos_path = os.path.join(test_path,'pos')
        test_neg_path = os.path.join(test_path,'neg')
        
        train_pos_files = list(map(lambda filename: os.path.join(train_pos_path, filename) ,os.listdir(train_pos_path)))  
        train_neg_files = list(map(lambda filename: os.path.join(train_neg_path, filename) ,os.listdir(train_neg_path )))
        test_pos_files = list(map(lambda filename: os.path.join(test_pos_path, filename) ,os.listdir(test_pos_path) ))
        test_neg_files = list(map(lambda filename: os.path.join(test_neg_path, filename) ,os.listdir(test_neg_path) ))
        
        dataset_train_pos = tf.data.TextLineDataset(train_pos_files)
        dataset_train_neg = tf.data.TextLineDataset(train_neg_files)
        dataset_test_pos = tf.data.TextLineDataset(test_pos_files)
        dataset_test_neg = tf.data.TextLineDataset(test_neg_files)

        dataset_train_pos = dataset_train_pos.take(n_train_over_2)    
        dataset_train_neg = dataset_train_neg.take(n_train_over_2)    
        dataset_test_pos = dataset_test_pos.take(n_train_over_2)     
        dataset_test_neg = dataset_test_neg.take(n_train_over_2)     
    
        n_validation = int(n_train * 0.6)
        n_test = int(n_train * 0.4)

        dataset_train_full = dataset_train_pos.concatenate(dataset_train_neg)
        text_vectorizer.adapt(dataset_train_full)
        embedding_layer = embedding_layer( output_dim = EMBEDDING_DIM, input_dim=len(text_vectorizer.get_vocabulary()) )

        dataset_train_pos_u = dataset_train_pos.map(lambda x: [x,0], num_parallel_calls=n_threads)
        dataset_train_neg_u = dataset_train_neg.map(lambda x: [x,1], num_parallel_calls=n_threads)
        train_ds = dataset_train_pos_u.concatenate(dataset_train_neg_u).shuffle(shuffle_buffer)

        dataset_test_pos_u = dataset_test_pos.map(lambda x: [x,0], num_parallel_calls=n_threads)
        dataset_test_neg_u = dataset_test_neg.map(lambda x: [x,1], num_parallel_calls=n_threads)
        dataset_test_full_u = dataset_test_pos_u.concatenate(dataset_test_neg_u)
        
        validation_ds = dataset_test_full_u.take(n_validation).shuffle(shuffle_buffer)
        test_ds = dataset_test_full_u.skip(n_validation).shuffle(shuffle_buffer)

    preprocessed_train_ds = preprocess_pipeline(train_ds, text_vectorizer, embedding_layer, n_batch)
    preprocessed_validation_ds = preprocess_pipeline(validation_ds, text_vectorizer, embedding_layer, n_batch)
    preprocessed_test_ds = preprocess_pipeline(test_ds, text_vectorizer, embedding_layer, n_batch)

    return preprocessed_train_ds, preprocessed_validation_ds, preprocessed_test_ds

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

train_ds, validation_ds, test_ds = get_datasets(os.path.join(WORKING_DIR, 'aclImdb'), use_tfds=True, debug=DEBUG_MODE)
    
input_ = tf.keras.layers.Input(shape=[EMBEDDING_DIM])
hidden1 = tf.keras.layers.Dense(EMBEDDING_DIM, activation='relu')(input_)
hidden2 = tf.keras.layers.Dense(550, activation='relu')(hidden1)
hidden3 = tf.keras.layers.Dense(550, activation='relu')(hidden2)
hidden4 = tf.keras.layers.Dense(10, activation='relu')(hidden3)
output = tf.keras.layers.Dense(1, activation='sigmoid')(hidden4)
model = tf.keras.Model( inputs=[input_], outputs=[output])
print(model.summary())
model.compile( loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
history = model.fit(train_ds, validation_data=validation_ds)
