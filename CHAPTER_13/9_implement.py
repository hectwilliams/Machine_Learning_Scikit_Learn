'''

Filename: 9_implement.py

Creator: Hector Williams 

Repo: https://github.com/hectwilliams/Machine_Learning_Scikit_Learn

Description: 

  Pulls MNIST and saves protobuf encoded binaries locally in ./store directory and trains a neural network with a custom Standardization Layer
'''

import tensorflow_datasets as tfds
import matplotlib.pyplot as plt 
import tensorflow as tf
import numpy as np 
import os 

BATCH_SIZE = 128
BATCH_ADAPT_SIZE = 10000
DEBUG_SIZE = 20000
STORE_BINARY = True

class CustomStandardizationMNIST(tf.keras.Layer):

  def adapt (self, img_samples):
    '''

      calculate mean and variance of 10,000 randomly sampled training set images (average the accumulation)

    '''
    img_samples = tf.cast( img_samples, (tf.float32) )
    
    self.mean_ = tf.math.reduce_mean(img_samples, axis = 0, keepdims=False)

    self.std_ = tf.math.reduce_std(img_samples, axis = 0, keepdims=False)

  def call(self, X):
    
    return  (X - self.mean_) / ( self.std_  + tf.keras.backend.epsilon()  ) 

  def compute_output_shape(self, batch_input_shape):

    return batch_input_shape
  
def build_model(training_set_ds):
  '''

    build neural network, adapt standardization layer

  '''
  standardizer = CustomStandardizationMNIST()
  
  z = ii = tf.keras.layers.Input(shape=(28,28,1,))

  z = tf.keras.layers.Lambda(lambda  some_image : tf.cast(some_image , tf.float32)) (z)

  z = standardizer(z)
  
  z = tf.keras.layers.Flatten()(z)

  for _ in tf.range(3):
    z = tf.keras.layers.Dense(50,activation= 'tanh') (z)

  oo = tf.keras.layers.Dense(10, activation=tf.keras.layers.Softmax())(z)

  model = tf.keras.Model(inputs=[ii], outputs=[oo])

  model.compile (loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.SGD(learning_rate=0.0034) , metrics=[ tf.keras.metrics.SparseCategoricalAccuracy()])
  
  for img_samples, label_samples in training_set_ds.take(BATCH_ADAPT_SIZE).shuffle(BATCH_ADAPT_SIZE).batch(BATCH_ADAPT_SIZE) :

    standardizer.adapt(img_samples)

  return model 

def proto_example_images( ds, id, set_name):
  '''

    using protobuf 'Example' class serialize batch instances to disk

  '''
  for image_tensor, label_tensor in ds:

    encode_image_tensor_to_jpeg = tf.io.encode_jpeg(image_tensor)

    example = tf.train.Example (

      features=tf.train.Features(

        feature = {

          'label': tf.train.Feature( int64_list=tf.train.Int64List(value=[label_tensor])),

          'img': tf.train.Feature( bytes_list=tf.train.BytesList ( value= [  encode_image_tensor_to_jpeg.numpy()  ] ) )

        }

      )

    )
    
    with tf.io.TFRecordWriter(f'store/{set_name}/mnist_tfrecord_b_{id}') as f:

      f.write(example.SerializeToString())

def proto_images_store_binary(batch):

  for i, (images, labels) in tf.data.Dataset.enumerate(batch):

    images_current_batch = tf.data.Dataset.from_tensor_slices(images)

    labels_current_batch = tf.data.Dataset.from_tensor_slices(labels)

    ds = tf.data.Dataset.zip(images_current_batch, labels_current_batch)

    proto_example_images(ds, i, batch.__name__)

def read_tfrecord_quick():

  feature_descr  = {
    "label": tf.io.FixedLenFeature([], tf.int64), 
    "img": tf.io.VarLenFeature(tf.string)
  }

  try:

    for ser_example in tf.data.TFRecordDataset([f'store/test/mnist_tfrecord_b_{0}']):

      parsed_file = tf.io.parse_single_example(ser_example, feature_descr)

      label = parsed_file['label']
      
      image_ = tf.sparse.to_dense(parsed_file['img'])

      image_decode = tf.io.decode_jpeg(image_[0])
      
      print(label)

      print(image_decode)

  except:

    print('record does not exist')
        
dataset = tfds.load(name='mnist')

mnist_train, mnist_test = dataset['train'], dataset['test'] # dataset houses list of dicts 

mnist_train = mnist_train.map(lambda record: (record['image'], record['label'] ) )

mnist_test = mnist_test.map(lambda record: (record['image'], record['label'] ) )

model = build_model(  mnist_train )  

mnist_val = mnist_train.skip(50000).cache().batch(32, drop_remainder= True)

mnist_train = mnist_train.take(50000).cache().batch(32, drop_remainder= True)

mnist_test =  mnist_test.batch(32, drop_remainder= True)

mnist_val.__name__ = 'validation'

mnist_test.__name__ = 'test'

mnist_train.__name__ = 'train'


if STORE_BINARY:

  dirnames = [ 'store/test', 'store/validation', 'store/train' ] 
  
  nested_list = [ os.listdir(d) for d in dirnames ]
  
  files = [item for sublist in nested_list 
        for item in sublist]
  
  for fpath in files:
    if os.path.isfile(fpath):
      os.remove(fpath)
    
  proto_images_store_binary(mnist_train) 

  proto_images_store_binary(mnist_test) 

  proto_images_store_binary(mnist_val) 
  
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('checkpoint_mnist.keras', save_best_only=True)

tensorboard_cb = tf.keras.callbacks.TensorBoard(os.path.join(os.getcwd(), "tensorboard_logs"))

model.fit(
  mnist_train,  
  epochs = 20, 
  validation_data = mnist_val,
  callbacks=[checkpoint_cb, tensorboard_cb]
)

