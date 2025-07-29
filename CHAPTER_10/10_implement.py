''''
  Train a deep MLP on the MNIST dataset. Find a good learning rate to start with. 
'''
import matplotlib.pyplot as plt 
import numpy as np 
import tensorflow as tf 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, Normalizer
import math

LR_EVAL_ITERATIONS = 20
FIND_LEARN_RATE = True

def find_good_lr(model):

  amplitude = np.zeros(LR_EVAL_ITERATIONS)

  lr_axis = np.zeros(LR_EVAL_ITERATIONS)
  
  lr = 10**-7

  min_lr = np.finfo(np.float64).max

  prev_lr = min_lr
  
  loss_ = prev_lr

  prev_loss = None

  curr_loss = None 


  plt.ion()
  
  for i in range(LR_EVAL_ITERATIONS):
    
    model.compile (loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(lr), metrics=['accuracy'])

    model.fit(
      X_train, 
      Y_train, 
      epochs=2, 
      validation_data=(X_val, Y_val),
    )

    prev_loss = loss_

    loss_, accuracy_ = model.evaluate(X_test, Y_test)

    curr_loss = loss_

    if math.isnan(curr_loss) or curr_loss >= prev_loss :
      
      return prev_lr
    
    else: 

      min_lr = lr
    
    prev_lr = lr

    lr = np.exp(np.log(10**6)/LR_EVAL_ITERATIONS) * lr

    amplitude[i] = loss_

    lr_axis[i] = lr

    plt.cla()
    plt.title('learning_rate')
    plt.semilogy(lr_axis[:i + 1], amplitude[:i + 1])
    plt.draw()
    plt.pause(2)

  return min_lr

(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()

X_train, X_val , Y_train, Y_val  = train_test_split(X_train, Y_train, test_size=0.20)
 
images = tf.keras.layers.Input(shape=(28,28))

flatten_images = tf.keras.layers.Flatten()(images)

flatten_images_norm = tf.keras.layers.Normalization()(flatten_images)

h1 = tf.keras.layers.Dense(100, activation='relu')(flatten_images_norm)

h2 = tf.keras.layers.Dense(100, activation='relu')(h1)

h3 = tf.keras.layers.Dense(100, activation='relu')(h2)

estimates = tf.keras.layers.Dense(10, activation='softmax')(h3)

model = tf.keras.Model(inputs=[images], outputs=[estimates])

'''
  summary 

  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ input_layer (InputLayer)             │ (None, 28, 28)              │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten (Flatten)                    │ (None, 784)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 100)                 │          78,500 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 100)                 │          10,100 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_2 (Dense)                      │ (None, 100)                 │          10,100 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_3 (Dense)                      │ (None, 10)                  │           1,010 │

'''

if FIND_LEARN_RATE:

  lr = find_good_lr(model)

else :

  lr = 0.00630957344480194

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('checkpoint_mnist.keras')

model.compile (loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(lr), metrics=['accuracy'])

model.fit(
  X_train, 
  Y_train, 
  epochs=20, 
  validation_data=(X_val, Y_val),
  callbacks=[checkpoint_cb]
)

loss_, accuracy_ = model.evaluate(X_test, Y_test)

print('eval loss', loss_)

print('eval accuracy', accuracy_)


