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


# DATASET (least 1000 images per class. )

dataset = tfds.load(name='mnist')

mnist_train, mnist_test = dataset['train'], dataset['test'] # dataset houses list of dicts

mnist_train = mnist_train.map(lambda record: (record['image'], record['label'] ))

mnist_test = mnist_test.map(lambda record: (record['image'], record['label'] ))

mnist_val = mnist_train.skip(50000).cache().batch(32, drop_remainder= True)

mnist_train = mnist_train.take(50000)

# mnist_train = mnist_train.batch(32, drop_remainder= True) #.prefetch(tf.data.experimental.AUTOTUNE)


# PREPROCESSING 

normalization = tf.keras.layers.Normalization()

normalization.adapt(mnist_train.take(10000).map(lambda X, Y: X))

mnist_train, length = expander(mnist_train)

mnist_train = augmenter(mnist_train, length)

mnist_train = mnist_train.batch(32, drop_remainder= True).prefetch(tf.data.experimental.AUTOTUNE)


# MODEL

z = ii = tf.keras.layers.Input(shape=(28,28,1,))

z = normalization(z)

z = tf.keras.layers.Flatten()(z)

oo = tf.keras.layers.Dense(10, activation=tf.keras.layers.Softmax())(z)

model = tf.keras.Model(inputs=[ii], outputs=[oo])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics= [tf.keras.metrics.SparseCategoricalAccuracy()])

model.fit(mnist_train, epochs=10)

model.evaluate(mnist_test)

