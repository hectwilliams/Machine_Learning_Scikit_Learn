'''

  Create custom normalization layer and verify output with keras' Normalization Layer 

'''

import matplotlib.pyplot as plt 

import tensorflow as tf

import numpy as np 

from functools import partial 

import math 

import time

import os 

import pickle 


class CustomNormalizationLayer(tf.keras.Layer):

  def __init__(self, units, **kwargs):
    
    super().__init__(**kwargs) 

    self.units = units 
    self.epsilon = tf.constant(0.0001)

  def build(self, batch_input_shape):

    self.kernel = self.add_weight(
      
      name = 'kernel',

      shape = batch_input_shape[-1:],

      initializer = tf.keras.initializers.Ones(),

      dtype = tf.float32
    )

    self.bias = self.add_weight(

      name = 'bias',

      shape = batch_input_shape[-1:],

      initializer = tf.keras.initializers.Zeros(),

      dtype = tf.float32

    )
    
    super().build(batch_input_shape) 

  def call(self, X):

    mean_vector, variance_vector = tf.nn.moments(X, axes=-1, keepdims=True)

    stddev_vector = tf.math.sqrt(variance_vector)
    
    compute_data = (X - mean_vector) / (stddev_vector+ self.epsilon) # normalize matrix 
    
    compute_data = tf.multiply(self.kernel, compute_data)  # element-wise multiply (kernel vector times each matrix row)

    compute_data = compute_data + self.bias # element-wise add (bias vector times each matrix row)

    return compute_data

X = tf.random.uniform((32,3))  # 3 channels, batch-32 

norm = CustomNormalizationLayer(4)

norm_keras = tf.keras.layers.LayerNormalization()

y_custom = norm(X)

y_keras = norm_keras(X)

ax = plt.subplot(111,projection='3d')

ax.scatter(*tf.transpose(y_custom), label='custom')

ax.scatter(*tf.transpose(y_keras), color='red', label='keras')
 
ax.axis('off')

plt.legend()

plt.show()
    
