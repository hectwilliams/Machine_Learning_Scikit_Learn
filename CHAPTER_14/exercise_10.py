'''
    Use transfer learning for large image classification

    a.
    Create a training set containing at least 100 images per class 

    b.
    Split it into a training set, validation set, and a test set 

    c.
    Build the input pipeline, including the appropriate preprocessing operations, and optionally add data augmentation.

    d.
    Fine tuned a pretrained model on this dataset  

'''

#!/usr/bin/env python3

import os 
import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np 

N_VALIDATION = 5000
MODLE_FILE_KERAS = os.path.join(os.getcwd(), 'my_model.keras')
CIFAR_CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
TRAIN_SIZE = 5000
RESNET50_IMG_DIM = [224, 224]

def preprocess(data_images, labels = np.array([])):

    img_resize = tf.image.resize(data_images, RESNET50_IMG_DIM)

    images_tensors = tf.keras.applications.resnet50.preprocess_input(img_resize * 255)

    ds = tf.data.Dataset.from_tensor_slices(images_tensors)
    
    if len(labels) != 0:

        ds = tf.data.Dataset.zip (ds, tf.data.Dataset.from_tensor_slices(labels))  
        ds = ds.map( lambda a,b: (a, b[0]), num_parallel_calls=3)

    return ds.batch(1).prefetch(1) 

def top_k (model, k, inputs, target):

    y_proba = model.predict(inputs)

    top_k = tf.keras.applications.resnet50.decode_predictions(y_proba, k)

    for image_index in range(2):

        print(f'Image #{image_index}')

        for class_id, name, proba in top_k[image_index]:

            print(f' {class_id} - {name:12s} {proba*100:.3f}%\t {target[image_index] }')

# Load CIFAR Data 

cifar10 = tf.keras.datasets.cifar10
(X_train_full, y_train_full), (X_test, y_test) = cifar10.load_data()

X_valid, X_train = X_train_full[:N_VALIDATION] / 255.0, X_train_full[N_VALIDATION:] / 255.0
y_valid, y_train = y_train_full[:N_VALIDATION], y_train_full[N_VALIDATION:]
X_test = X_test / 255.0

if os.path.exists(MODLE_FILE_KERAS):
    raise Exception('pre-trained model already exists in working directory; delete .keras file if changes have been made to the training dataset')

train_dataset = preprocess(X_train, y_train)
validation_dataset = preprocess(X_valid, y_valid)
test_dataset = preprocess(X_test)

base_model_resnet = tf.keras.applications.ResNet50(weights='imagenet', include_top=False) 

model = tf.keras.Sequential([
    tf.keras.Input(shape = (224,224,3,)),
    base_model_resnet, 
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(len(CIFAR_CLASSES) , activation='softmax'),
])

# show DNN Architecture 

print(base_model_resnet.summary())
print(model.summary())

# Freeze base layers 

for layer in base_model_resnet.layers:
    layer.trainable = False 

model.compile( loss = 'sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(learning_rate=0.2, momentum=0.9, decay=0.01), metrics=['accuracy'] )

# Callbacks  
 
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(MODLE_FILE_KERAS, save_best_only=True)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)

# Train 

history = model.fit(train_dataset, epochs=2, validation_data=validation_dataset, callbacks=[checkpoint_cb, early_stopping_cb])

