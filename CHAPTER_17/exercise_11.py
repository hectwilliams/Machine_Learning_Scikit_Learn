#!/usr/bin/env python3 

"""
    Train a DCGAN to tackle the image dataset of your choice, and use it to generate images.
    Add experience replay and see it this helps. 
"""

import os
import sys 
import numpy as np 
import tensorflow as tf 
from exercise_9 import SaveModelsCallback, process_serialize_img_data, AddDenseBlock
from exercise_10 import parse_autoencoder, Sampling
import matplotlib.pyplot as plt

# Download CFAR-100 into current working directory: see https://www.cs.toronto.edu/~kriz/cifar.html 
# unzip cifar-100-python.tar.gz (unzipped directory should be cifar-100-python)
DS_DIR = os.path.join(os.getcwd(), 'cifar-100-python')
BATCH_SIZE = 32
N_EPOCHS = 5000
CODING_SIZE =  128 * 4 * 4 
N_REPLAYS = 10

@tf.keras.utils.register_keras_serializable(package="HtronLayers1")
class Sampling (tf.keras.Layer):
    def call(self, inputs):
        mean, log_var = inputs 
        random_normal_vector = tf.random.uniform(tf.shape(log_var))
        std_dev_vector =  tf.math.exp(log_var/2)
        return random_normal_vector * std_dev_vector + mean

def process_serialize_img_data(serialize_data_norm):
    xplit = tf.split( tf.cast(serialize_data_norm , tf.float32), 3) 
    for i in range(3) :
        xplit[i] = tf.reshape(xplit[i] ,(32,32)) 
    return tf.stack(xplit, axis = -1)  

def train_gan(gan, ds, batch_size, codings_size):
    generator, discriminator = gan.layers 
    
    try:
        generator = tf.keras.models.load_model(os.path.join(os.getcwd(), 'ex11_generator.keras'))
        discriminator = tf.keras.models.load_model(os.path.join(os.getcwd(), 'ex11_discriminator.keras'))
        gan = tf.keras.models.load_model(os.path.join(os.getcwd(), 'ex11_gan.keras'))
    except :
        pass

    for index in range(1, N_EPOCHS + 1):

        loss = None

        for batch in ds:
            
            # train discriminator ( I know what is real or fake, I trained myself on real checks)
            noise = tf.random.normal(shape=[batch_size, codings_size])

            gen_images = generator(noise)
            
            x_fake_and_real = tf.concat([gen_images, batch], axis=0)    

            y1 = tf.constant( [[0.0]] * batch_size + [[1.0]] * batch_size )

            discriminator.trainable = True 
            loss = discriminator.train_on_batch(x_fake_and_real, y1)

            # train gan (Generator is trained and learned to make realistic checks)
            noise = tf.random.normal(shape=[batch_size, codings_size])
            y2 = tf.constant( [[1.0]] * batch_size )  # counterfeit checks are real, believe me discriminator!
            discriminator.trainable = False 
            gan.train_on_batch(noise, y2)
        
        print(f'epoch {index}/{N_EPOCHS}')
        print(f'avg loss {loss}')
              
        gan.save(os.path.join(os.getcwd(), 'ex11_gan.keras'))
        generator.save(os.path.join(os.getcwd(), 'ex11_generator.keras'))
        discriminator.save(os.path.join(os.getcwd(), 'ex11_discriminator.keras'))
    
    # test dan 
    noise = tf.random.normal(shape=(1, codings_size))
    test_image = generator(noise)
    plt.imshow(test_image[0])
    plt.show()

if __name__ == '__main__':

    if sys.argv.__len__() < 2 :
        raise ValueError('Are you training or inferring?')

    ds = parse_autoencoder('train', False).map(process_serialize_img_data).batch(BATCH_SIZE, drop_remainder= True).prefetch(1) 

    if sys.argv[1] == 'train':

        z_input_generator = tf.keras.layers.Input(shape=[CODING_SIZE])
        z = tf.keras.layers.Reshape([4, 4, 128])(z_input_generator)
        z = tf.keras.layers.Conv2DTranspose( 64, kernel_size = 5, strides=2, padding='same', activation='selu')(z)
        z = tf.keras.layers.Conv2DTranspose( 32, kernel_size = 5, strides=2, padding='same', activation='selu')(z)
        z_out = tf.keras.layers.Conv2DTranspose( 3, kernel_size = 2, strides=2, padding='same', activation='sigmoid')(z)
        model_generator = tf.keras.Model(inputs=[z_input_generator], outputs=[z_out])

        z_input_discriminator = tf.keras.layers.Input(shape=[32, 32, 3])
        z = tf.keras.layers.Conv2D( 32, kernel_size = 5, strides=2, padding='same', activation=tf.keras.activations.leaky_relu)(z_input_discriminator)
        z = tf.keras.layers.Conv2D( 64, kernel_size = 4, strides=2, padding='same', activation=tf.keras.activations.leaky_relu)(z)
        z = tf.keras.layers.Conv2D( 128, kernel_size = 4, strides=2, padding='same', activation=tf.keras.activations.leaky_relu)(z)
        z = tf.keras.layers.Flatten()(z)
        z_out = tf.keras.layers.Dense(1, activation='sigmoid')(z)
        model_discriminator = tf.keras.Model(inputs=[z_input_discriminator], outputs=[z_out])
        
        gen_images = model_generator(z_input_generator)
        discrm_class = model_discriminator( gen_images  )
        dcgan = tf.keras.Sequential([model_generator, model_discriminator]) 
        
        model_discriminator.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer='rmsprop')
        # discriminator model NOT trainable by dcgan model 
        model_discriminator.trainable = False
        dcgan.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer='rmsprop')

        train_gan(dcgan, ds, BATCH_SIZE, CODING_SIZE)
    
    elif sys.argv[1] == 'infer':
        generator = tf.keras.models.load_model(os.path.join(os.getcwd(), 'ex11_generator.keras'))
        noise = tf.random.normal(shape=(1, CODING_SIZE))
        test_image = generator(noise)
        plt.imshow(test_image[0])
        plt.show()
