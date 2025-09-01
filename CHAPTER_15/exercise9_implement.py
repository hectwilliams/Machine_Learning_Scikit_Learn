'''
  
  Train a classification model for the SketchRNN dataset 

  Note: Model has not been officially trained. This solution shows the architecture, preprocessing, and parsing required to train such a model. This problem gets exponentially larger with additional classes (e.g. dog, MonaLisa, etc)

  -H.W.

'''

import tensorflow as tf
import tensorflow_datasets as tfds
import requests
import os
import numpy as np
import asyncio
import matplotlib.pyplot as plt
import json 

CATEGORY = ['cat.full', 'airplane.full', 'basketball.full']
PLOT_SKETCH = False 
BATCH_SIZE = 32

async def get_sketchRNN(category):
  
  """

    Request datasets from Quick Draw API and save them locally.

    Endpoint: https://console.cloud.google.com/storage/browser/quickdraw_dataset/sketchrnn

  """

  try:

    for category_name in CATEGORY:

      url = f'https://storage.googleapis.com/quickdraw_dataset/sketchrnn/{category_name}.npz'

      ds_filepath = f'{category_name}.npz'

      if not ds_filepath is os.listdir(os.getcwd()):

        response = requests.get(url, stream=True)

        response.raise_for_status() # bad staus check

        with open(ds_filepath, 'wb') as f:

          for chunk in response.iter_content(chunk_size=8192):

            f.write(chunk)

  except requests.exceptions.RequestException as e:

    print(f'Error downloading file: {e}')

async def short_subplot(name_ds):
  
  """

    Plot 3 figures housing 49 sketches of cat, airplanes, and basketballs

  """

  k = 7

  fig, ax = plt.subplots(k, k)

  fig.set_size_inches(10, 10)

  fig.set_edgecolor('black')
  fig.set_facecolor('white')
  fig.set_linewidth(10)

  data = np.load(name_ds, 'r+', allow_pickle=True, encoding='bytes')

  data_train = data['train']

  print(data_train.shape)
  for idx, sketchData in enumerate(data_train[:k*k]):

    r = idx // 7

    c = (idx % 7)

    ax[r][c].axis('off')

    ax[r][c].set_aspect('equal')

    ax[r][c].set_xlim([-500,500])

    ax[r][c].set_ylim([-500, 500])

    xx = [0,0]

    yy = [0,0]

    for deltax, deltay, new_pos in sketchData:

      if new_pos == 0:

        xx[1] = xx[1] + deltax

        yy[1] = yy[1] + deltay

        ax[r][c].plot( xx, yy )

        xx[0] = xx[1]

        yy[0] = yy[1]

      elif new_pos == 1:

        xx[0] =  xx[1]

        yy[0] =  yy[1]

        xx[1] = xx[0] + deltax

        yy[1] = yy[0]  + deltay

        break

  return ax

async def wait_for_sketch_npz():
  
  np.random.seed(0)

  while 1:

    count = 0

    for category_name in CATEGORY:

      ds_filepath = f'{category_name}.npz'

      count += int(ds_filepath in os.listdir(os.getcwd()))

    if count == len(CATEGORY):

      break

    await asyncio.sleep(0.5)

  cat_ds = f'cat.full.npz'
  
  airplane_ds = f'airplane.full.npz'
  
  basketball_ds = f'basketball.full.npz'

  if PLOT_SKETCH:
    
    a = await short_subplot(cat_ds)
    
    b = await short_subplot(airplane_ds)
    
    c = await short_subplot(basketball_ds)

    if a is not None and b is not None and c is not None:

      plt.setp(zip(a, b, c), xticks=[], yticks=[])

  # load downloaded quickdraw datasets to RAM (.npz files)
  
  data_cat = np.load(cat_ds, 'r', allow_pickle=True, encoding='bytes')
  
  data_airplane = np.load(airplane_ds, 'r', allow_pickle=True, encoding='bytes')
  
  data_basketball = np.load(basketball_ds, 'r', allow_pickle=True, encoding='bytes')

  for name in ['train', 'test', 'valid']:

    # preprocess datasets (padding)

    packed_cat = data_cat[name]
    pad_npz(packed_cat)

    packed_airplane = data_airplane[name]
    pad_npz(packed_airplane)
    
    packed_basketball = data_basketball[name]
    pad_npz(packed_basketball)

    label_cat = np.zeros( dtype=np.int32, shape = len(packed_cat) )

    label_airplane = np.zeros( dtype=np.int32, shape = len(packed_airplane) ) + 1

    label_basketball = np.zeros( dtype=np.int32, shape = len(packed_basketball) ) + 2

    Y = np.concat((label_cat, label_airplane, label_basketball))  #  0 - cat, 1 - airplane, 2 - basketball

    X = np.concat((packed_cat, packed_airplane, packed_basketball))

    # shuffle dataset

    indices = np.arange( len(X) )
    
    for i in range(10):
      
      np.random.shuffle(indices)

    X = (X[indices])

    Y = (Y[indices]).astype(np.int32)

    # write proto 

    with tf.io.TFRecordWriter(f'data_{name}.tfrecord') as writer:

      for id in range(len(X)):
        
        seq = X[id]

        bytes_seq = tf.io.serialize_tensor(tf.convert_to_tensor(seq))

        label = Y[id]

        record_example = tf.train.Example(
          features=tf.train.Features(
            feature={
              'sketch': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes_seq.numpy()])),
              'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }))
        
        writer.write(record_example.SerializeToString())

    with open(f'data_{name}.tfrecord.json', 'w') as json_file:

      json.dump(dict(length= len(Y)), json_file)

  return True

def pad_npz(packed_data):
  """

    Left pad the time series so all series data have the same length.
    
  """

  # LEFT PADDING (TIME SERIES LENGTH = 130)

  max_seq = 256

  pen_no_op = np.array([0,0,1])[None, ...] # 1 x 3

  for i in range(len(packed_data)):

    seq = packed_data[i]

    n_padding = max_seq - len(seq)

    if n_padding > 0:

      pad = np.repeat(pen_no_op, n_padding, axis=0)

      packed_data[i] = np.concatenate((pad, packed_data[i]))


def get_model():

  """

    Architecture to predict time series. Because we only care to predict on the last step of a series the RNN block preceding the output layer DOES NOT return sequences. 

  """

  z = ii = tf.keras.layers.Input(shape= (None,3) , name ='input')

  z = tf.keras.layers.SimpleRNN(units=30, return_sequences=True) (z)

  z = tf.keras.layers.SimpleRNN(30)(z)
  
  z = oo = tf.keras.layers.Dense(3, activation='softmax') (z)

  model = tf.keras.Model(inputs=[ii], outputs=[oo])

  return model 

@tf.function
def train_me(n_steps, n_steps_valid, n_epochs, ds_train, ds_valid, model, train_length, valid_length, batch_size, mean_loss, mean_loss_valid,  optimizer, loss_fn, mean_train_loss_series, valid_loss_series):
  
  """

    Custom training set 

  """

  for epoch in tf.range(n_epochs):

    for X, Y in ds_train.take(n_steps):
      
      X = tf.reshape(X, (32, 130, 3))

      with tf.GradientTape() as tape:
        
        Y_pred = model(X, Training = True)

        loss_ret = loss_fn(Y, Y_pred)
        
        mean_loss(loss_fn(Y,Y_pred))

        gradients = tape.gradient(loss_ret, model.trainable_variables)

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    mean_train_loss_series[epoch].assign(mean_loss.result())

    for X_batch, Y_batch in ds_valid.take(n_steps_valid): 

      X_batch = tf.reshape(X_batch, (32, 130, 3))

      Y_pred = model(X_batch, Training = False)

      mean_loss_valid (loss_fn(Y_batch, Y_pred))

    valid_loss_series[epoch].assign(mean_loss_valid.result())
    
    mean_loss_valid.reset_state()

    mean_loss.reset_state()

def parse_local_tfrecord(name = 'train'):
  """

    Read locally saved tfrecord files.

  """
  json_data = json.load(open(f'data_{name}.tfrecord.json'))
    
  length = json_data['length']

  feature_description = {
        
      'sketch': tf.io.FixedLenFeature([], tf.string),

      'label': tf.io.FixedLenFeature([], tf.int64)

  }  
  
  ds = tf.data.TFRecordDataset(f'data_{name}.tfrecord') 

  ds = ds.map(lambda x: tf.io.parse_single_example(x, feature_description))

  ds = ds.map(lambda record: (tf.io.parse_tensor(record['sketch'], out_type=tf.int64) , record['label']) ) 
  
  ds = ds.shuffle(buffer_size = length, seed=32).repeat(1).batch(32, drop_remainder=False)

  return ds, length
    
if __name__ == '__main__':

  ds = None 
  
  status = True
  
  if 'data_train.tfrecord' not in os.listdir(os.getcwd()):

    status = False
    
    await asyncio.create_task(get_sketchRNN(CATEGORY))

    status = await asyncio.create_task(wait_for_sketch_npz())

  if status: 

    ds_train, ds_train_length  = parse_local_tfrecord()

    ds_test, ds_test_length = parse_local_tfrecord('test')

    ds_valid, ds_valid_length = parse_local_tfrecord('valid')

    model = get_model()
    
    mean_loss = tf.keras.metrics.Mean(name='loss')

    mean_loss_valid = tf.keras.metrics.Mean(name='loss_valid') 

    loss_fun = tf.keras.losses.SparseCategoricalCrossentropy()

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    n_steps = 25 #int(ds_train_length // BATCH_SIZE)

    n_steps_valid = int(ds_valid_length // BATCH_SIZE)

    n_epochs = 10

    avg_train_loss_series = tf.Variable(tf.zeros(n_epochs))
    
    valid_loss_series = tf.Variable(tf.zeros(n_epochs))
    
    train_me(n_steps, n_steps_valid, n_epochs, ds_train, ds_valid, model, ds_train_length,ds_valid_length, BATCH_SIZE, mean_loss, mean_loss_valid, optimizer, loss_fun, avg_train_loss_series, valid_loss_series)

    plt.plot(avg_train_loss_series, label='train', color='black')

    plt.scatter(tf.range(n_epochs),avg_train_loss_series ,  c= 'black', s = 10, marker='*')
    
    plt.plot(valid_loss_series, label='valid', color='blue')

    plt.scatter(tf.range(n_epochs),valid_loss_series ,  c= 'blue', s = 10, marker='*')

    plt.legend()

    for x_test, y_test in ds_test.take(1):

      x_test = tf.reshape(x_test, (32, 130, 3))

      y_pred = model(x_test, Training = False)

      accuracy = tf.keras.metrics.SparseCategoricalAccuracy(y_test, y_pred )

      print(y_pred)

      print(y_test)

      print('Accuracy-\t', accuracy)

    plt.show()



