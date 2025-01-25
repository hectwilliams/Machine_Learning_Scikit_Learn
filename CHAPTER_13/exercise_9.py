#!/usr/bin/env python3

'''

Load the Fasion MNIST dataset' split it into a training set, a validation set, and a test set; shuffle the training set; and save each dataset to miltiple TFRecords files.
Each record should be a serialized Example  protobuf with two features; the serialized image (use tf.io.serialize_tensor() to serialize each image), and the label.
Then use tf.data to create an efficient dataset for each set. Finally, use a Keras model to train these datasets, including a preprocessing layer to standardsize each input feature.
Try to make the input pipeline as efficient as possible, using TensorBoard to visualize profiling data.

'''

from tensorflow import keras 
import tensorflow as tf  
from functools import partial
import os 
import numpy as np
import csv
import sys 
import time 
from tensorflow.train import BytesList, FloatList, Int64List, Feature, Features, Example

VALIDATION_SIZE = 5000
RECORDS_PER_FILE = 200
N_IMAGE_PIXELS_ROW = 28
N_IMAGE_PIXELS_COL = 28
RESET_MODEL = True
USE_TENSORBOARD = True
WR_TFRECORDS_TO_DISK = True 
PROTOBUF_SCHEMAS = {
    'debug': { 'label': tf.io.FixedLenFeature([], tf.int64, default_value=[0]), 'img': tf.io.FixedLenFeature( [ N_IMAGE_PIXELS_ROW*N_IMAGE_PIXELS_COL], tf.float32, default_value=[0.0 for i in range(N_IMAGE_PIXELS_ROW*N_IMAGE_PIXELS_COL)  ]) , },
    'default': { 'id': tf.io.FixedLenFeature([], tf.int64, default_value=[0]),  'label': tf.io.FixedLenFeature([], tf.int64, default_value=[0]), 'img': tf.io.FixedLenFeature( [], tf.string, default_value=b'ERROR') }
}

SCHEMA =  PROTOBUF_SCHEMAS['default']

class PrintValTrainRatioCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self,logs):
        print(f'\t\t\tTRAIN BEGIN')
    def on_train_end(self, logs):
        print(f'\t\t\tTRAIN END')
    def on_epoch_begin(self, epoch, logs):
        print(f'\t\t\tEPOCH BEGIN')
    def on_epoch_end(self, epoch, logs):
        val_loss = logs['val_loss']
        train_loss = logs['loss']
        print(f"\n val/train {val_loss/train_loss :.2f}\t EPOCH END")

def normalize_images_data(data_train, data_test, data_label_train, validation_size= None):
    X_valid, X_train , X_test, y_valid, y_train = data_train[:validation_size]/255.0, data_train[validation_size:] / 255.0, data_test/255.0, data_label_train[:validation_size], data_label_train[validation_size:], 
    print( f'  X_TRAIN:\t{len(X_train)} \n VALID_SIZE:\t{len(X_valid)}\nTEST_SIZE:\t{len(X_test)}' )
    return X_valid, X_train , X_test, y_valid, y_train , len(X_train), len(X_valid), len(X_test)

def encode_to_disk(img_batch:np.ndarray, img_target, name_id, dir, id_table) :
    wr_count = 0
    # length = img_batch.shape[-1] *img_batch.shape[-1]
    # csv_dir = os.path.join(dir, 'csv')
    # csv_file_path = os.path.join(csv_dir, f'{name_id}.csv')
    # headers = [f'pixel{i}' for i in range( length  ) ] + ['label']
    # with open(csv_file_path, mode='w') as f:
    #     csv_writer = csv.DictWriter(f, headers)
    #     csv_writer.writeheader()
    #     for img, label in zip(img_batch, img_target):
    #         img_flatten_label = img.reshape(-1).tolist() + [label]
    #         csv_writer.writerow({header:img_flatten_label[index]  for index,header in enumerate(headers)})
    
    tfrecord_dir = os.path.join(dir, 'tfrecord')
    tfs_file_path = os.path.join(tfrecord_dir, f'{name_id}.tfrecord')
    with tf.io.TFRecordWriter(tfs_file_path) as writer: 
        for img, label in zip(img_batch, img_target):
            serialize_img = tf.io.serialize_tensor( img ).numpy()
            if serialize_img not in id_table:
                id_table[serialize_img] = len(id_table)
            encode_example = Example(features = Features(feature = {  'img': Feature(bytes_list=BytesList(value=[ serialize_img ] )), 'label': Feature(int64_list=Int64List(value=[label]))  , }  )) # encode_example = Example(features = Features(feature = {'img': Feature(float_list=FloatList(value=img.reshape(-1).tolist())) , 'label': Feature(int64_list=Int64List(value=[label]))} )  )
            wr_count+= 1
            serialized_example = encode_example.SerializeToString()
            writer.write(serialized_example)
    assert(wr_count == RECORDS_PER_FILE)
    
def clear_files(*dirs):
    for dir in dirs: 
        for file in os.scandir(dir):
            if file.name.endswith(".csv") or file.name.endswith(".tfrecord") or file.name.endswith(".record"):
                os.unlink(file.path)

def make_dirs(*dirs):
    for current_dir in dirs:
        if not os.path.exists( current_dir ):
            os.makedirs(current_dir)

def tfrecord_reader_dataset (file_paths, table , n_file_readers = 5, shuffle_buffer_size =1000, n_preprocess_threads = 5, batch_size = 32 ):
    
    dataset_tfrecords = tf.data.TFRecordDataset(file_paths, num_parallel_reads=n_file_readers) # interleaved records of files read (i.e. parrelel reads) 
    dataset = dataset_tfrecords.map(  preprocess_10  )  
    dataset = dataset.map(lambda X,Y: ( (X, table.lookup(X)),  Y)  )
    dataset = dataset.shuffle(shuffle_buffer_size) 
    dataset = dataset.batch(batch_size, num_parallel_calls=n_preprocess_threads).prefetch(1)
    return dataset

def preprocess_10(proto_record):
    proto_dict = tf.io.parse_single_example(proto_record, SCHEMA) 
    return  proto_dict['img'] , proto_dict['label'] 

def dict_to_static_vocab_table(some_dict) :
    vocab = list(some_dict.keys()) # bytes data stored in keys
    indices = tf.range(len(vocab), dtype=tf.int64)
    init_table = tf.lookup.KeyValueTensorInitializer(vocab, indices) 
    return tf.lookup.StaticVocabularyTable( init_table,  num_oov_buckets)

def wr_serial_tfrecords(train_data, label_data, dir, csv_dir, tfrecord_dir, table_dict):
    make_dirs(dir, csv_dir, tfrecord_dir)
    clear_files( csv_dir, tfrecord_dir )
    n_tfrecord_per_file = len(train_data) // RECORDS_PER_FILE 
    data_split = np.split(train_data, n_tfrecord_per_file )
    label_split = np.split(label_data, n_tfrecord_per_file)
    for index, (batch, batch_label ) in enumerate( zip(data_split, label_split)):
        encode_to_disk(batch, batch_label, index, dir, table_dict)
    
def build_model(model_path, num_oov_buckets,train_len, use_best_prev_model):
    if use_best_prev_model:
        model_ = tf.keras.models.load_model(model_path)
    else:
        img_input_STRING_DO_NOT_USE = tf.keras.layers.Input(shape=[], name='serial_img')
        id_input = tf.keras.layers.Input(shape=[], name='id_img') 
        embed = tf.keras.layers.Embedding(input_dim=train_len + num_oov_buckets, output_dim=3, name='Embedded')(id_input)
        dense1 = tf.keras.layers.Dense(200, activation='relu') (embed)
        dense2= tf.keras.layers.Dense(100, activation='relu') (dense1)
        output = tf.keras.layers.Dense(10, activation='softmax') (dense2)
        model_ = tf.keras.Model(inputs=[img_input_STRING_DO_NOT_USE , id_input  ], outputs=[output])
        print(model_.summary())
    return model_ 

model_save_path = os.path.join( os.getcwd(), 'model.keras')

model_path_to_architecture =  os.path.join( os.getcwd(), 'model_view_arch.png')

checkpoint_epoch_end_cb = tf.keras.callbacks.ModelCheckpoint(model_save_path)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

tensorboard_cb = []

fashion_mnist = tf.keras.datasets.fashion_mnist

# Remote Data
(X_train_all, y_train_all), (X_test_all, y_test_all) = fashion_mnist.load_data() # Shape: (60000, 28, 28) , dtype: uint8 , Type: numpy array

# limit size for debug 
CURRENT_TRAIN_SIZE= 20000
X_train_all = X_train_all[:CURRENT_TRAIN_SIZE]
y_train_all = y_train_all[:CURRENT_TRAIN_SIZE]

# normalize_images_data = partial(normalize_images_data_)
X_valid, X_train , X_test, y_valid, y_train, train_size, valid_size, test_size = normalize_images_data(X_train_all, X_test_all, y_train_all, VALIDATION_SIZE) 
y_test = y_test_all

# Write Serial Records to Local disk
train_dir = os.path.join( os.getcwd(), 'train_dir')
train_csv_dir = os.path.join( os.getcwd(), train_dir, 'csv')
train_tfrecord_dir = os.path.join( os.getcwd(), train_dir, 'tfrecord')

validate_dir = os.path.join( os.getcwd(), 'validate_dir')
validate_csv_dir = os.path.join( os.getcwd(), validate_dir, 'csv')
validate_tfrecord_dir = os.path.join( os.getcwd(), validate_dir, 'tfrecord')

test_dir = os.path.join( os.getcwd(), 'test_dir')
test_csv_dir = os.path.join( os.getcwd(), test_dir, 'csv')
test_tfrecord_dir = os.path.join( os.getcwd(), test_dir, 'tfrecord')

table_img_id_train = dict() 
table_img_id_valid = dict() 
table_img_id_test = dict() 
num_oov_buckets = 2

if WR_TFRECORDS_TO_DISK:
    wr_serial_tfrecords(X_train, y_train, train_dir, train_csv_dir, train_tfrecord_dir, table_img_id_train)
    wr_serial_tfrecords(X_valid, y_valid, validate_dir, validate_csv_dir, validate_tfrecord_dir, table_img_id_valid)
    wr_serial_tfrecords(X_test, y_test, test_dir, test_csv_dir, test_tfrecord_dir, table_img_id_test)

# build lut table for training data 
train_table = dict_to_static_vocab_table(table_img_id_train) 
valid_table = dict_to_static_vocab_table(table_img_id_valid) 
test_table = dict_to_static_vocab_table(table_img_id_test) 

file_paths_train = list(map(lambda name: train_tfrecord_dir + '/' + name ,os.listdir(train_tfrecord_dir) ))
train_dataset = tfrecord_reader_dataset(file_paths_train, train_table , n_file_readers=3)

file_paths_valid = list(map(lambda name: validate_tfrecord_dir + '/' + name ,os.listdir( validate_tfrecord_dir ) ))
valid_dataset = tfrecord_reader_dataset(file_paths_valid,valid_table, n_file_readers=3)

file_paths_test = list(map(lambda name: test_tfrecord_dir + '/' + name ,os.listdir( test_tfrecord_dir ) ))
test_dataset = tfrecord_reader_dataset(file_paths_test,test_table,  n_file_readers=3)

# train model
model = build_model(model_save_path, num_oov_buckets, train_size, False)
tf.keras.utils.plot_model(model, to_file=model_path_to_architecture, show_shapes=True, show_layer_names=True )
model.compile( loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

if USE_TENSORBOARD:
    curr_run_id = time.strftime('run_%Y_%m_%d-%H_%M_%S')
    root_log_dir = os.path.join(os.getcwd(),'tensorboard_log')
    run_log_dir = os.path.join(root_log_dir, curr_run_id)
    tensorboard_cb = tf.keras.callbacks.TensorBoard(run_log_dir)

history = model.fit(train_dataset, epochs=10, validation_data=valid_dataset, callbacks=[checkpoint_epoch_end_cb, early_stopping_cb, PrintValTrainRatioCallback()] + [tensorboard_cb])


