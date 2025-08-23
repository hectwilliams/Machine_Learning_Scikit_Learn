'''
  
  Transfer learning (using xception model) to train mnist 

'''



import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

def show_class_numbers(mnist_train):

  histo = {}

  for x, y in mnist_train:

    if y.numpy() in histo:

        histo[y.numpy()] += 1

    else:

        histo[y.numpy()] = 1

  print(histo)


def expander(ds: tf.data.Dataset, repeat_f = 10):

  # window 32 items ( one window per item )

  length = len(ds)

  if repeat_f <=1:
    length

  ds = ds.window(1, shift=1, drop_remainder=True) # window 5 images

  ds = ds.repeat(repeat_f)

  ds = tf.data.Dataset.zip(ds, tf.data.Dataset.range(len(ds)))

  ds = tf.data.Dataset.range(length).map(  lambda window_id:

                                       ds.filter(lambda window, id: tf.math.mod(id , length) == window_id    )  )
  ds = ds.flat_map(lambda x: x)

  ds = ds.map(lambda xy,iden : xy)

  ds = ds.flat_map(lambda X, Y: tf.data.Dataset.zip ( X, Y ))

  return ds , length * repeat_f

def augmenter(ds: tf.data.Dataset, length ):

  ids = tf.data.Dataset.range(length)

  ds = ds.flat_map(lambda x, y:  tf.data.Dataset.zip( tf.data.Dataset.from_tensors(x), tf.data.Dataset.from_tensors(y) , ids)  ) # flatten and zip

  ds = ds.map( lambda x, y, id :  ( (augment_image(x), y) if id % 10 == 0  else  (augment_image(x), y) )   )

  return ds

def augment_image(image):

  image = tf.cond(tf.random.normal(shape=()) > 0.2, lambda: tf.pad(image, [[10,0], [10,0], [0,0]] )[ 0:28, 0:28, : ] , lambda: image)

  image = tf.cond(tf.random.normal(shape=()) > 0.2, lambda: tf.image.rot90(image) , lambda: image)

  image = tf.cond(tf.random.normal(shape=()) > 0.2, lambda: tf.image.flip_left_right(image) , lambda: image)

  image = tf.cond(tf.random.normal(shape=()) > 0.2, lambda: tf.image.random_brightness(image, 0.4) , lambda: image)

  return image

def preprocess_xception(X, Y):

  X_resize = tf.image.resize(X, (224,224))

  X_resize = tf.keras.applications.xception.preprocess_input(X_resize)
    
  return X_resize, Y

def preprocess_check(X, Y):

  if tf.shape(X)[-1] == 1:
    
    X = tf.image.grayscale_to_rgb(X)

  return X, Y 

# * DATASET * (least 1000 images per class. )

dataset = tfds.load(name='mnist')

mnist_train, mnist_test = dataset['train'], dataset['test'] # dataset houses list of dicts

mnist_train = mnist_train.map(lambda record: (record['image'], record['label'] ))

mnist_test = mnist_test.map(lambda record: (record['image'], record['label'] ))

mnist_val = mnist_train.skip(50000)

mnist_train = mnist_train.take(50000)

# * PREPROCESSING *

mnist_test = mnist_test.map(preprocess_xception)

mnist_val = mnist_val.map(preprocess_xception)

mnist_train, length = expander(mnist_train)

mnist_train = augmenter(mnist_train, length)

mnist_train = mnist_train.map(preprocess_xception)

mnist_train = mnist_train.map(preprocess_check)

mnist_val = mnist_val.map(preprocess_check)

mnist_test = mnist_test.map(preprocess_check)

mnist_train = mnist_train.batch(32, drop_remainder= True).prefetch(tf.data.experimental.AUTOTUNE)

mnist_test = mnist_test.batch(32, drop_remainder= True).prefetch(tf.data.experimental.AUTOTUNE)

mnist_val = mnist_val.batch(32, drop_remainder= True).prefetch(tf.data.experimental.AUTOTUNE)

# * MODEL *

base_model = tf.keras.applications.xception.Xception(weights='imagenet', include_top=False)

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)

prediction_layer = tf.keras.layers.Dense(units=10, activation='softmax') (global_average_layer)

model = tf.keras.Model(inputs=base_model.input, outputs = prediction_layer)

# freeze model 

for layer in base_model.layers:

  layer.trainable = False 

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.03), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics= [tf.keras.metrics.SparseCategoricalAccuracy()])

history = model.fit(mnist_train, epochs=10, validation_data=mnist_val)

model.evaluate(mnist_test)

# unfreeze model 

for layer in base_model.layers:

  layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics= [tf.keras.metrics.SparseCategoricalAccuracy()])

history = model.fit(mnist_train, epochs=10)

# evaluate 

model.evaluate(mnist_test)



