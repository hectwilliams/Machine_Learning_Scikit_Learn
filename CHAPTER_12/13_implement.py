'''
  Train model using custom training loop to tackle the Fashio MNIST dataset
'''

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt 

from functools import partial 

import tensorflow as tf

import numpy as np 

import pickle 

import math 

import time

import os 

@tf.function
def random_batch(X: tf.Tensor, Y:tf.Tensor, batch_size: tf.Tensor =tf.constant(32)  ):
  
  indices = tf.keras.random.shuffle(tf.range(len(X)), seed=32)[:batch_size]
  
  return tf.gather(X, indices=indices), tf.gather(Y, indices=indices)

# TRAINING DATA  

(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()

X_train, X_val , Y_train, Y_val  = train_test_split(X_train, Y_train, test_size=0.20)

X_train_tf = tf.convert_to_tensor(X_train)

Y_train_tf = tf.convert_to_tensor(Y_train)

X_val_tf = tf.convert_to_tensor(X_val)

Y_val_tf = tf.convert_to_tensor(Y_val)

# MODEL SECTION 

z = ii = tf.keras.layers.Input(shape=(28,28), name='images_in')

z = tf.keras.layers.Flatten()(z)

for i in range(3):
  z = tf.keras.layers.BatchNormalization()(z)
  z = tf.keras.layers.Dense(10,activation= tf.keras.layers.ELU(), kernel_initializer=tf.keras.initializers.HeNormal()) (z)

oo = tf.keras.layers.Dense(10, activation=tf.keras.layers.Softmax(), kernel_initializer= tf.keras.initializers.GlorotNormal() )(z)

model = tf.keras.Model(inputs=[ii], outputs=[oo])

print(model.summary())

# TRAINING PARAMETERS SECTION 

mean_tr_loss = tf.keras.metrics.Mean()

mean_tr_accuracy = tf.keras.metrics.Mean()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

accuracy_fn =  tf.keras.metrics.SparseCategoricalAccuracy()

batch_size = tf.constant(32)

n_epochs = tf.constant(10)

n_steps =  tf.constant(len(X_train)//batch_size )

optimizerSGD = tf.keras.optimizers.Nadam(learning_rate=0.00032)

optimizerNadam = tf.keras.optimizers.SGD(learning_rate=0.0032)

# TRAINING SECTION 

for epoch in range(1, n_epochs + 1):

  for step in range(1, n_steps + 1):

    x_batch, y_batch = random_batch(X_train_tf, Y_train_tf)
    
    with tf.GradientTape() as tape:
      
      y_pred = model(x_batch, training=True) 
      
      training_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))

      gradients = tape.gradient(training_loss, model.trainable_variables)

      # lower layers optimized with Nadam at a lower rate
      lower_gradients = gradients[ : len(gradients)//2]
      lower_trainables = model.trainable_variables[ : len(gradients)//2]
      
      optimizerNadam.apply_gradients(zip( lower_gradients, lower_trainables))

      # upper layers optimized with SGD at one-tenth of Nadam learning-rate
      upper_gradients = gradients[ len(gradients)//2 : ]
      upper_trainables = model.trainable_variables[ len(gradients)//2 : ]

      optimizerSGD.apply_gradients(zip(upper_gradients, upper_trainables))

    mean_tr_loss(training_loss)
    
    mean_tr_accuracy( accuracy_fn(y_batch, y_pred) ) 

    metric_loss_list = [ "{:.3f}".format( m.result() ) for m in [ mean_tr_loss] + [ mean_tr_accuracy] ] 

    print(f'|epoch={epoch} |step={step} |{" - ".join( metric_loss_list )}')
  
  mean_tr_loss.reset_state()

  mean_tr_accuracy.reset_state()
