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

# DATASET WITH at least 1000 images per class

dataset = tfds.load(name='mnist')

mnist_train, mnist_test = dataset['train'], dataset['test'] # dataset houses list of dicts

mnist_train = mnist_train.map(lambda record: (record['image'], record['label'] ))

mnist_test = mnist_test.map(lambda record: (record['image'], record['label'] ))

mnist_val = mnist_train.skip(50000).cache().batch(32, drop_remainder= True)

# show_class_numbers(mnist_train)

mnist_train = mnist_train.take(1000)#.batch(32).cache()

# mnist_train = mnist_train.batch(32, drop_remainder= True) #.prefetch(tf.data.experimental.AUTOTUNE)

# CUSTOM INPUT

class PREPROCESSING(tf.keras.layers.Layer):

  def __init__(self, channel_first = False, **kwargs):

    self.channel_first = channel_first

    super().__init__(**kwargs)

  # @tf.function # AutoGraph convert if else statements to tf.cond within graph
  def augment_image(self, image):

    image = tf.cond(tf.random.normal(shape=()) > 0.5, lambda: tf.pad(image, [[10,0], [10,0], [0,0]] )[ 0:28, 0:28, : ] , lambda: image)

    image = tf.cond(tf.random.normal(shape=()) > 0.5, lambda: tf.image.rot90(image) , lambda: image)

    image = tf.cond(tf.random.normal(shape=()) > 0.5, lambda: tf.image.flip_left_right(image) , lambda: image)

    image = tf.cond(tf.random.normal(shape=()) > 0.5, lambda: tf.image.random_brightness(image, 0.4) , lambda: image)

    return image

  def call(self, X):

    # CAST

    X = tf.cast(X, tf.int32)

    # NORMALIZE

    X = X / tf.reduce_max(X)

    # ADD AUGMENTED IMAGES

    additional_images = tf.map_fn(self.augment_image, X)

    X = tf.concat((X, additional_images), axis=0)

    return X

  def build(self, batch_input_shape):

    self.batch_count = batch_input_shape[0]

    return super().build(batch_input_shape)

  def compute_output_shape(self, batch_input_shape):

    array = tf.Variable(tf.ones(shape=(len(batch_input_shape))))

    array[0].assign(2)

    return batch_input_shape

    return tf.multiply(batch_input_shape, array)

    batch_input_shape = tf.multiply(batch_input_shape, array)


  def get_config(self):

    return super().get_config()

# PREPROCESSING EXPANDER 

def expander(ds: tf.data.Dataset, repeat_f = 10):

  # window 32 items ( one window per item )

  length = len(ds)

  ds = ds.window(1, shift=1, drop_remainder=True) # window 5 images

  ds = ds.repeat(repeat_f) 
  
  ds = tf.data.Dataset.zip(ds, tf.data.Dataset.range(len(ds)))

  ds = tf.data.Dataset.range(length).map(  lambda window_id: 
                                       
                                       ds.filter(lambda window, id: tf.math.mod(id , length) == window_id    )  )

  ds = ds.flat_map(lambda x: x)

  ds = ds.map(lambda xy,iden : xy) 
  
  ds = ds.flat_map(lambda X, Y: tf.data.Dataset.zip ( X, Y ))

  return ds 

# PREPROCESSING IMAGE AUGMENT

def augmenter(ds: tf.data.Dataset, repeat_f = 10):

mnist_train = expander(mnist_train)
# mnist_train = mnist_train.map(expander)

# count = 0

for x,y in mnist_train.batch(32):

  print(x, y)
  assert(0)

assert(0)

# MODEL

z = ii = tf.keras.layers.Input(shape=(28,28,1,))

z = PREPROCESSING(channel_first=False)(z)

z = tf.keras.layers.Flatten()(z)

oo = tf.keras.layers.Dense(10, activation=tf.keras.layers.Softmax())(z)

model = tf.keras.Model(inputs=[ii], outputs=[oo])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics= [tf.keras.metrics.SparseCategoricalAccuracy()])

model.fit(mnist_train, epochs=10)

# model.evaluate(mnist_test)

