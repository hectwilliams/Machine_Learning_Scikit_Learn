
import matplotlib.pyplot as plt 

import tensorflow as tf

import numpy as np 

from functools import partial 

import math 

import time

import os 

import pickle 

LR_EVAL_ITERATIONS = 100

USE_LOADABLE_MODEL = True

TRAIN_MC = False

TRAIN_MC_SIZE = 500 

def find_good_lr(model, X_train, Y_train, X_val, Y_val, X_test, Y_test):

  amplitude = np.zeros(LR_EVAL_ITERATIONS)

  lr_axis = np.zeros(LR_EVAL_ITERATIONS)
  
  lr = 0.0003019951720402016

  min_lr = np.finfo(np.float64).max

  prev_lr = min_lr
  
  loss_ = prev_lr

  prev_loss = None

  curr_loss = None 

  for i in range(LR_EVAL_ITERATIONS):
    
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.Nadam(lr))

    model.fit(
      X_train[:1000], 
      Y_train[:1000], 
      epochs=2, 
      validation_data=(X_val, Y_val),
    )

    prev_loss = loss_

    loss_= model.evaluate(X_test, Y_test)

    curr_loss = loss_

    if math.isnan(curr_loss) or curr_loss >= prev_loss :
      
      return prev_lr
    
    else: 

      min_lr = lr
    
    prev_lr = lr

    lr = np.exp(np.log(10**6)/LR_EVAL_ITERATIONS) * lr

    amplitude[i] = loss_

    lr_axis[i] = lr
  
  return min_lr

def get_tensorNewDir():

  return os.path.join(os.getcwd(), "logs")

(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.cifar10.load_data()

X_train, X_val = X_train[:40000], X_train[-10000:]

Y_train, Y_val = Y_train[:40000], Y_train[-10000:]

class MyLearningRateCallback(tf.keras.callbacks.Callback):
  
  def on_batch_end(self, batch_index, logs=None):
    
    row  = self.custom_epoch

    col = batch_index % 1250

    x_curr_pos = row * 1250 + col 
    
    if x_curr_pos < 12500:

      self.model.optimizer.lr = x_curr_pos *  2e-07 + 0.00030
    
    elif x_curr_pos < 24900:

      self.model.optimizer.lr = x_curr_pos *  -2.1774193548387098e-07 + 0.00571

    else:
      
      self.model.optimizer.lr = x_curr_pos * -2.5822580645161253e-06 +0.06458645161290313
  
  def on_epoch_begin(self, epoch, logs=None):
    
    self.custom_epoch = epoch


hiddenDense = partial(tf.keras.layers.Dense, units=100, activation='selu', kernel_initializer='lecun_normal')

z = ii =  tf.keras.layers.Input(shape=(32,32,3), name='images_in')

z = tf.keras.layers.Flatten()(z)

for i in range(20):
  z = hiddenDense()(z)
  z = tf.keras.layers.AlphaDropout(rate=0.15, seed=32)(z) 

oo = tf.keras.layers.Dense(10, activation='softmax', kernel_initializer='glorot_normal')(z)

model = tf.keras.Model(inputs=[ii], outputs=[oo])

if USE_LOADABLE_MODEL:

  model = tf.keras.models.load_model('checkpoint_mnist.keras')

else:

  lr = find_good_lr(model, X_train, Y_train, X_val, Y_val, X_test, Y_test)

  model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.Nadam(lr))
  
  checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('checkpoint_mnist.keras', save_best_only=True)

  tensorboard_cb = tf.keras.callbacks.TensorBoard(get_tensorNewDir())

  custom_lr_scheduler_cb = MyLearningRateCallback() 
  
  model.fit (
    X_train, 
    Y_train, 
    epochs = 20, 
    validation_data=( (X_val, Y_val) ),   
    callbacks=[checkpoint_cb, tensorboard_cb, custom_lr_scheduler_cb]
  )

if not os.path.exists('mc_pickle.pkl') or TRAIN_MC:
  
  y_probas = np.stack( [model(X_test, training=True) for _ in range(TRAIN_MC_SIZE)] ) # training argument enables dropout for each predict
  
  with open ('mc_pickle.pkl', 'wb') as file:

    pickle.dump({'prob':y_probas}, file)

file = open("mc_pickle.pkl",'rb')

y_probas = pickle.load(file)['prob'] # TRAIN_MC_SIZE monte carlo states , 10000 instances per state, 10 features 

file.close()

model_pred_test = model.predict(X_test)

mc_model_pred_test = np.sum(y_probas, axis=0) / TRAIN_MC_SIZE

std_dev = y_probas.std(axis=0)

metric_sparse_categorical_cross = tf.keras.metrics.SparseCategoricalCrossentropy()

pred_model_accuracy = metric_sparse_categorical_cross(Y_test, model_pred_test)

pred_mc_model_accuracy = metric_sparse_categorical_cross(Y_test, mc_model_pred_test)

print(pred_model_accuracy, pred_mc_model_accuracy) # horrible accuracy :( 