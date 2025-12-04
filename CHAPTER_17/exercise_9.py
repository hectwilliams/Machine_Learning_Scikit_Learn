"""

Name: exercise_9.py

Description:

  Create denoise autoencoder. After training autoencoder, train classifer on top 500 reconstructed dataset samples(i.e images).

  The best 500 reconstructed images are further used to train a classifier.

  Steps:

    1) Enter a directory path in PREFIX variable 

    2) Set TRAIN_MODE = 0, and run script

      ** Autoencoder trained. Script will stop training when loss is acceptable ( .npy files generated)

    3) Set TRAIN_MODE = 2 and run script

      ** Classifier DNN trained

      ** Classifier does not converge :( 

"""

# %load_ext tensorboard
# %tensorboard --logdir logs/fit

import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
import shutil

CREATE_RECORDS = False

N_EPOCHS = 600

PREFIX = '~/tmp_autoencoder/' # assume linux 

CFAR_TRAIN_10 = PREFIX + 'cifar10_train.tfrecord'

CFAR_TEST_10 = PREFIX + 'cifar10_test.tfrecord'

WEIGHTS_FILENAME = PREFIX + 'autoencoder.weights.h5'

WEIGHTS_FILENAME_NO_PRETRAIN = PREFIX + 'autoencoder_non_pretrain.weights.h5'

WEIGHTS_FILENAME_PRETRAIN = PREFIX + 'autoencoder_pretrain.weights.h5'

WEIGHTS_FILENAME_PRETRAIN_GOLD = PREFIX + 'gold_autoencoder.weights.h5'

TOP_IMAGES = PREFIX + 'top_imagesx'

TOP_INDICES = PREFIX + 'top_indice'

TOP_LABELS = PREFIX + 'top_labels'

USE_GOLD_PRETRAIN = True

MODEL_FILENAME = 'autoencoder.keras'

LOG_DIR = 'logs/fit/' + datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S")

BATCH_SIZE = 4

INFER = False

TRAIN_MODE = 2 #- autoencoder, 1 - non pretrained model, 2 - pretrained model

LEARNING_RATE =  0.000116014 if TRAIN_MODE == 2 else  0.00061100116014

LEARNING_DECAY_STEPS = 10000

LEARNING_DECAY_RATE = 0.94

REPEAT = 4

CFAR_SIZE = 10

DEPTHS =  [1, 1, 1 , 1, 1, 1, 1]

USE_PRETRAIN_LAYER = True

DROPOUT_RATE = 0.7

DEEP_DN = 2

CODING_SHAPE = (4,4,256)

CODING_SIZE = CODING_SHAPE[0] * CODING_SHAPE[1] * CODING_SHAPE[2]

def unpickle(file):
  import pickle
  with open(file, 'rb') as fo:
      dict = pickle.load(fo, encoding='bytes')
  return dict

def serial_to_rbg(serial):
  red = serial[:1024]
  green = serial[1024:2048]
  blue = serial[2048:]
  red = red.reshape(32,32) / 255.0
  green = green.reshape(32,32) / 255.0
  blue = blue.reshape(32,32) / 255.0
  return np.dstack((red, green, blue))

class Encoder(tf.keras.Model):

  def __init__(self, **kwargs):

    super().__init__(**kwargs)

    self.conv2D_INIT = tf.keras.layers.Conv2D(64, 2, activation='selu', padding='SAME', kernel_initializer= tf.keras.initializers.GlorotUniform(seed=32))

    self.conv2D_16s = [tf.keras.layers.Conv2D(32, 3, activation='selu', padding='SAME', kernel_initializer= tf.keras.initializers.GlorotUniform(seed=32)) for i in range(DEPTHS[0])]

    self.conv2D_32s = [tf.keras.layers.Conv2D(32, 3, activation='selu', padding='SAME', kernel_initializer= tf.keras.initializers.GlorotUniform(seed=32)) for i in range(DEPTHS[1])]

    self.conv2D_64s = [tf.keras.layers.Conv2D(64, 3, activation='selu', padding='SAME', kernel_initializer= tf.keras.initializers.GlorotUniform(seed=32)) for i in range(DEPTHS[2])]

    self.conv2D_128s = [tf.keras.layers.Conv2D(128, 3, activation='selu', padding='SAME', kernel_initializer= tf.keras.initializers.GlorotUniform(seed=32)) for i in range(DEPTHS[3])]

    self.conv2D_256s = [tf.keras.layers.Conv2D(256, 3, activation='selu', padding='SAME', kernel_initializer= tf.keras.initializers.GlorotUniform(seed=32)) for i in range(DEPTHS[4])]

    # self.conv2D_512s = [tf.keras.layers.Conv2D(512, 3, activation='selu', padding='SAME', kernel_initializer= tf.keras.initializers.GlorotUniform(seed=32)) for i in range(DEPTHS[5])]

    # self.codings = tf.keras.layers.Dense(CODING_SIZE, activation='selu', kernel_initializer= tf.keras.initializers.GlorotUniform(seed=32))

    self.max_pooling2D = tf.keras.layers.MaxPool2D(2)

    self.add_guassian_noise_layer = tf.keras.layers.Lambda(lambda x: tf.random.normal(shape=( 32, 32, 3), stddev=0.3,  seed=32, ) + x)

    self.glayer = tf.keras.layers.GaussianNoise(stddev=0.2, seed=32)

    self.flatten = tf.keras.layers.Flatten()

    self.reg = tf.keras.layers.ActivityRegularization(l1=1e-3)

    self.global_avg = tf.keras.layers.GlobalAveragePooling2D()

  def call(self, x):

    x = self.glayer(x)

    x = self.conv2D_INIT(x)

    for layer in self.conv2D_16s :

      x = layer(x)

    for layer in self.conv2D_32s :

      x = layer(x)

    for layer in self.conv2D_64s :

      x = layer(x)

    x = self.max_pooling2D(x) # 16 x 16

    for layer in self.conv2D_128s :

      x = layer(x)

    x = self.max_pooling2D(x) # 8 x 8 x 128

    for layer in self.conv2D_256s :

      x = layer(x)

    x = self.max_pooling2D(x) # 4 x 4 x 256

    x = self.flatten(x) # CODING_SIZE

    x = self.reg(x) # L1 Regularizer

    return x

  def get_config(self):

    base_config = super().get_config()

    return {

        **base_config,

      }

class Decoder(tf.keras.Model):

  def __init__(self, **kwargs):

    super().__init__(**kwargs)

    self.reshape = tf.keras.layers.Reshape(CODING_SHAPE)

    self.transpose_conv2D_512 = tf.keras.layers.Conv2DTranspose(512, 3, activation='selu', strides = 1, padding='SAME', kernel_initializer= tf.keras.initializers.GlorotUniform(seed=32))

    self.transpose_conv2D_256 = tf.keras.layers.Conv2DTranspose(256, 3, activation='selu', strides = 1, padding='SAME', kernel_initializer= tf.keras.initializers.GlorotUniform(seed=32))

    self.transpose_conv2D_128 = tf.keras.layers.Conv2DTranspose(128, 3, activation='selu', strides = 2, padding='SAME', kernel_initializer= tf.keras.initializers.GlorotUniform(seed=32))

    self.transpose_conv2D_64 = tf.keras.layers.Conv2DTranspose(64, 3, activation='selu', strides = 1, padding='SAME', kernel_initializer= tf.keras.initializers.GlorotUniform(seed=32))

    self.transpose_conv2D_32 = tf.keras.layers.Conv2DTranspose(32, 3, activation='selu', strides = 1, padding='SAME', kernel_initializer= tf.keras.initializers.GlorotUniform(seed=32))

    self.transpose_conv2D_16 = tf.keras.layers.Conv2DTranspose(16, 3, activation='selu', strides = 1, padding='SAME', kernel_initializer= tf.keras.initializers.GlorotUniform(seed=32))

    self.transpose_conv2D_8 = tf.keras.layers.Conv2DTranspose(8, 3, activation='selu', strides = 2, padding='SAME', kernel_initializer= tf.keras.initializers.GlorotUniform(seed=32))

    self.transpose_conv2D_4 = tf.keras.layers.Conv2DTranspose(4, 3, activation='selu', strides = 1, padding='SAME', kernel_initializer= tf.keras.initializers.GlorotUniform(seed=32))

    self.transpose_conv2D_3 = tf.keras.layers.Conv2DTranspose(3, 3, activation='sigmoid', strides = 2, padding='SAME', dtype=tf.float64, kernel_initializer= tf.keras.initializers.GlorotUniform(seed=32))


  def call(self, x):

    x = self.reshape(x)

    # x = self.transpose_conv2D_512(x)

    # x = self.transpose_conv2D_256(x)

    x = self.transpose_conv2D_128(x)

    x = self.transpose_conv2D_64(x)

    x = self.transpose_conv2D_32(x)

    x = self.transpose_conv2D_16(x)

    x = self.transpose_conv2D_8(x)

    x = self.transpose_conv2D_4(x)

    x = self.transpose_conv2D_3(x)

    return x

  def get_config(self):

    base_config = super().get_config()

    return {

        **base_config,

      }

class ClassifierDNN (tf.keras.Model):

  def __init__(self, enc_layer = None, dec_layer = None,  **kwargs):

    super().__init__(**kwargs)

    self.flatten = tf.keras.layers.Flatten()

    self.dense_4096s = [tf.keras.layers.Dense( 32, activation='selu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=32) ) for _ in range(DEEP_DN)]

    self.dense_2048s = [tf.keras.layers.Dense( 32, activation='selu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=32) ) for _ in range(DEEP_DN)]

    self.dense_1024s = [tf.keras.layers.Dense( 32, activation='selu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=32) ) for _ in range(DEEP_DN)]

    self.dense_32 = tf.keras.layers.Dense( 32, activation='selu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=32) )

    self.dense_32s = [tf.keras.layers.Dense( 32, activation='selu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=32) ) for _ in range(DEEP_DN)]

    self.dense_10s = [tf.keras.layers.Dense( 32, activation='selu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=32) ) for _ in range(DEEP_DN)]

    self.dense_10 = tf.keras.layers.Dense( CFAR_SIZE, activation='softmax')

    self.enc_layer = enc_layer

    self.dec_layer = dec_layer

    self.global_avg_2D = tf.keras.layers.GlobalAveragePooling2D()

    self.reg = tf.keras.layers.ActivityRegularization(l2=1e-3)

    self.dropout = tf.keras.layers.Dropout(0.4)

  def call(self, x):

    if self.enc_layer:

      z = self.enc_layer(x)

      z = self.dec_layer(z)

      z = self.global_avg_2D(z)

      z = self.reg(z)

      # z = self.flatten(z)

    else:

      z = self.flatten(x)

    for index, dense_layer in enumerate(self.dense_2048s):

      z = dense_layer(z)

      break

    for index, dense_layer in enumerate(self.dense_1024s):

      z = dense_layer(z)

      break

    z = self.dropout(z)

    z = self.dense_10(z)

    return z

  def get_config(self):

    base_config = super().get_config()

    return {

        **base_config,

        'enc_layer' : self.enc_layer

      }



class Top500ImagesCB(tf.keras.callbacks.Callback):

  def __init__(self, train_dataset):

    super().__init__()

    self.train_dataset = train_dataset

  def on_epoch_end(self, epoch, logs= None):

      current_loss = logs.get("loss")

      n_range = 500;

      if current_loss <= 0.0050 and not os.path.exists(TOP_IMAGES + '.npy'):

        top_ = [[] for i in range(n_range)]

        for batch_index, (img_x, label_y) in enumerate(self.train_dataset):

          recon = self.model(img_x)

          mse_val_batch = tf.keras.losses.MSE( tf.reshape(recon, (BATCH_SIZE,-1) ), tf.reshape(img_x, (BATCH_SIZE,-1)) )

          for list_index, mse_value in enumerate(mse_val_batch):

            for k in range(n_range):

              obj = top_[k]

              if len(obj) and obj[2] >= mse_value:

                top_ = top_[0 : k] + [[batch_index, list_index, mse_value, recon[list_index] , label_y[0]]] + top_[ k + 1 : ]

                break

            for k in range(n_range):

              if top_[k] == []:

                top_[k] = [batch_index, list_index, mse_value, recon[list_index] , label_y[0] ]

                break

        count = 0

        for a in top_ :

          if a :

            count += 1

        images = []

        indices = []

        labels = []

        for data in top_:

          if data:

            images.append(data[3])

            labels.append(data[4])

            indices.append( (data[0] * BATCH_SIZE) + data[1] )

        np.save(TOP_IMAGES, images)

        np.save(TOP_INDICES, indices)

        np.save(TOP_LABELS, labels)

        assert(0)

def build_model_ae():

  encoder_input_tensor = tf.keras.layers.Input(shape=(32,32,3), name='IMAGES_IN')

  encoder = Encoder( name='ENCODER_MODULE')

  encoder_output_tensor = encoder(encoder_input_tensor)

  decoder = Decoder(name='DECODER_MODULE')

  decoder_output_tensor = decoder(encoder_output_tensor)

  autoencoder_model = tf.keras.Model(inputs=[encoder_input_tensor], outputs=[decoder_output_tensor])

  print(autoencoder_model.summary())

  return autoencoder_model

def build_model_non_pretrain():

  input_tensor = tf.keras.layers.Input(shape=(32,32,3), name='IMAGES_IN')

  dnn = ClassifierDNN( name='CLASSIFIER_MODULE')

  output_tensor = dnn(input_tensor)

  model = tf.keras.Model(inputs=[input_tensor], outputs=[output_tensor])

  print(model.summary())

  return model

def build_model_pretrain():

  model_ae = build_model_ae()

  model_ae.load_weights(WEIGHTS_FILENAME)

  encoder_ae = model_ae.layers[1]

  decoder_ae = model_ae.layers[2]

  # freeze encoder layer

  counter = 0

  for index , layer in enumerate(encoder_ae.layers):

    if layer.built == True:

      if 'conv2d' in layer.name:

        if counter < 1:

          layer.trainable = False

        else:

          layer.trainable = True

        counter += 1

    print(f' {index} - {layer} {layer.trainable} ' )

  # freeze decoder layer (bottom 3)

  counter = 0

  for index , layer in enumerate(decoder_ae.layers):

    if layer.built == True:

      if 'conv2d' in layer.name:

        if counter < 0:

          layer.trainable = False

        else:

          layer.trainable = True

        counter += 1

    print(f' {index} - {layer} {layer.trainable} ' )


  print(model_ae.summary())

  print(encoder_ae.summary())

  print(decoder_ae.summary())

  dnn = ClassifierDNN(enc_layer=encoder_ae, dec_layer=decoder_ae, name='CLASSIFIER_MODULE')

  input_tensor = tf.keras.layers.Input(shape=(32,32,3), name='IMAGES_IN')

  output_tensor = dnn(input_tensor)

  model = tf.keras.Model(inputs=[input_tensor], outputs=[output_tensor])

  print(model.summary())

  return model

def wr_tf_records():

  test_batch = unpickle('data_batch_1')

  test_batch_x = test_batch[b'data']

  test_batch_packed = unpickle('test_batch')

  test_batch_y = test_batch_packed[b'labels']

  test_batch_x = test_batch_packed[b'data']

  d = np.zeros(shape=(50000, 32,32,3))

  l = np.zeros(shape=(50000))

  t = np.zeros(shape=(10000))

  t_label = np.zeros(shape=(10000))

  data = None

  label = None

  with tf.io.TFRecordWriter(CFAR_TRAIN_10) as writer:

    for file_id in range(1, 6):

      batch_packed = unpickle('data_batch_' + str(file_id))

      batch_packed_x = batch_packed[b'data']

      batch_packed_y = batch_packed[b'labels']

      for i in range(10000):

        i = 10000 * (file_id - 1)  + i

        data = serial_to_rbg(batch_packed_x[i % 10000])

        label = batch_packed_y[i % 10000]

        example_train = tf.train.Example(features=tf.train.Features(feature={

          "serial_img": tf.train.Feature( bytes_list=tf.train.BytesList( value = [ tf.io.serialize_tensor(data).numpy() ] )  ),

          "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[  tf.cast( label, tf.int64)  ]))

        }))

        writer.write(example_train.SerializeToString())


  with tf.io.TFRecordWriter(CFAR_TEST_10) as writer:

    test_batch_packed = unpickle('test_batch')

    for i in range(10000):

      data = serial_to_rbg (test_batch_packed[b'data'][i])

      label = test_batch_packed[b'labels'][i]

      example_train = tf.train.Example(features=tf.train.Features(feature={

        "serial_img": tf.train.Feature( bytes_list=tf.train.BytesList( value = [ tf.io.serialize_tensor(data).numpy() ] )  ),

        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[  tf.cast( label, tf.int64)   ]))

      }))

      writer.write(example_train.SerializeToString())

def is_tfrecord_corrupted(tfrecord_file):
    try:
        for record in tf.data.TFRecordDataset(tfrecord_file):
            # Attempt to parse the record
            _ = tf.train.Example.FromString(record.numpy())
    except tf.errors.DataLossError as e:
        print(f"DataLossError encountered: {e}")
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return True
    return False

if CREATE_RECORDS:

  wr_tf_records()

# load dataset
feature_description = {

  'serial_img': tf.io.FixedLenFeature([], tf.string),

  'label': tf.io.FixedLenFeature([], tf.int64)

}

print(is_tfrecord_corrupted(CFAR_TRAIN_10))

print(is_tfrecord_corrupted(CFAR_TEST_10))

train_dataset = tf.data.TFRecordDataset( CFAR_TRAIN_10 )

train_dataset = train_dataset.map(lambda x: tf.io.parse_single_example(x, feature_description)).map(lambda record: (tf.io.parse_tensor(record['serial_img'], out_type=tf.float64), int(record['label'] )) )

test_dataset = tf.data.TFRecordDataset(CFAR_TEST_10)

test_dataset = test_dataset.map(lambda x: tf.io.parse_single_example(x, feature_description)).map(lambda record: (tf.io.parse_tensor(record['serial_img'], out_type=tf.float64), int(record['label']) ) )

train_dataset = train_dataset.map(lambda x, y: ( tf.reshape(x, (32,32,3)), y[None, ...] )    )

train_dataset_xy = train_dataset

train_dataset_test = train_dataset

test_dataset = test_dataset.map(lambda x, y: ( tf.reshape(x, (32,32,3)), y[None, ...] )    )

train_ae_ds = train_dataset.map(lambda x, y: ( x, x )    )

test_ae_ds = test_dataset.map(lambda x, y: ( x, x )    )

train_dataset = train_dataset.batch(BATCH_SIZE,drop_remainder= True).prefetch(tf.data.experimental.AUTOTUNE)

test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder= True).prefetch(tf.data.experimental.AUTOTUNE)

train_ae_ds = train_ae_ds.batch(BATCH_SIZE, drop_remainder= True, num_parallel_calls = 2).prefetch(tf.data.experimental.AUTOTUNE)

test_ae_ds = test_ae_ds.batch(BATCH_SIZE, drop_remainder= True).prefetch(tf.data.experimental.AUTOTUNE)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=300, restore_best_weights=True)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
      initial_learning_rate=LEARNING_RATE,
      decay_steps=LEARNING_DECAY_STEPS,
      decay_rate=LEARNING_DECAY_RATE
  )

# DIR CREATE

if not os.path.isdir(PREFIX):

  os.mkdir(PREFIX)

# BEST IMAGES

images_ds_best = None

ds_500 = None

if os.path.exists(TOP_IMAGES + '.npy') and os.path.exists(TOP_INDICES + '.npy') :

  images = np.load(TOP_IMAGES + '.npy', allow_pickle=True)

  indices = np.load(TOP_INDICES + '.npy', allow_pickle=True)

  labels = np.load(TOP_LABELS + '.npy', allow_pickle=True)

  images_ds_best = tf.data.Dataset.from_tensor_slices(images)

  indices_ds = tf.data.Dataset.from_tensor_slices(indices)

  labels_ds = tf.data.Dataset.from_tensor_slices(labels)

  count = 0

if images_ds_best:

  ds_500 = tf.data.Dataset.zip(images_ds_best, labels_ds, indices_ds)

if ds_500:

  ds_500 = tf.data.Dataset.zip(images_ds_best, labels_ds)

  ds_500 = ds_500.map(lambda x,y : (x, y[0]))

  ds_500 = ds_500.batch(BATCH_SIZE, drop_remainder=True)


# TRAIN IMAGES

if TRAIN_MODE == 0 :

  autoencoder_model = build_model_ae()

  if os.path.exists(WEIGHTS_FILENAME):

    autoencoder_model.load_weights(WEIGHTS_FILENAME)

    autoencoder_model.compile (loss= "huber", optimizer=tf.keras.optimizers.Adam(learning_rate = lr_schedule))

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(WEIGHTS_FILENAME, save_best_only=True, save_weights_only=True)

    top_evaluate_cb = Top500ImagesCB(train_dataset_xy.batch(BATCH_SIZE))

    autoencoder_model.fit(

      train_ae_ds,

      epochs=N_EPOCHS,

      validation_data=test_ae_ds,

      callbacks = [

        early_stopping_cb,

        checkpoint_cb,

        # top_evaluate_cb
      ]

    )

elif TRAIN_MODE == 1:

  model = build_model_non_pretrain()

  if os.path.exists(WEIGHTS_FILENAME_NO_PRETRAIN):

      model.load_weights(WEIGHTS_FILENAME_NO_PRETRAIN )

  model.compile (loss= tf.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule))

  checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(WEIGHTS_FILENAME_NO_PRETRAIN, save_best_only=True, save_weights_only=True)

  model.fit( train_dataset.unbatch().take(500).repeat(REPEAT).batch(BATCH_SIZE, drop_remainder= True), epochs=N_EPOCHS, validation_data= test_dataset.unbatch().take(500).batch(BATCH_SIZE, drop_remainder= True), callbacks = [early_stopping_cb, checkpoint_cb] )

elif TRAIN_MODE == 2:

  model = build_model_pretrain()

  if os.path.exists(WEIGHTS_FILENAME_PRETRAIN):

    model.load_weights(WEIGHTS_FILENAME_PRETRAIN )

  model.compile (loss= tf.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.RMSprop(learning_rate=lr_schedule))

  checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(WEIGHTS_FILENAME_PRETRAIN, save_best_only=True, save_weights_only=True)

  model.fit( ds_500.unbatch().repeat(REPEAT).shuffle(REPEAT* 500).batch(BATCH_SIZE, drop_remainder= True).prefetch(tf.data.AUTOTUNE) , epochs=N_EPOCHS,

              validation_data= test_dataset.unbatch().take(2000).map(lambda x,y: (x, y[0]) ).batch(BATCH_SIZE, drop_remainder= True), callbacks = [early_stopping_cb, checkpoint_cb])


