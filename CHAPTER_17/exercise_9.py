'''
    Q. Try using a denoising autoencodder to pretrain an image classifier. 

    A. I will train CIFAR - 100 dataset

    Q. Split dataset into training and test set. Train a deep denoising autoencoder on full training set

    -> A.  Training is on-going. Current loss = 0.0616. My target is 0.03 where the reconstruction of images is good.

    Q. Build a classification DNN, reusing the lower layers of the autoencoder. Train it using only 500 images from the training set. Does it perform better with or without pretraining?

    A.  Pretraining helps the model converge to lower loss 
 
'''

#!/usr/bin/env python3
import os
import numpy as np
import pickle
import tensorflow as tf 
import sys 
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE 

# Download CFAR-100 into current working directory: see https://www.cs.toronto.edu/~kriz/cifar.html 
# unzip cifar-100-python.tar.gz (unzipped directory should be cifar-100-python)
DS_DIR = os.path.join(os.getcwd(), 'cifar-100-python') 
N_255 = 255.0
N_EPOCHS = 3000
BATCH_SIZE = 2
NORMALIZE_INPUT = True 

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def process_serialize_img_data(serialize_data_norm):
    xplit = np.array_split( tf.cast(serialize_data_norm  * (tf.constant(1.0,dtype=tf.float32) if  NORMALIZE_INPUT else 1.0) , tf.float32), 3)
    d = np.array(list(zip(*xplit)))
    d = np.reshape(d, (32,32,3))
    return d

def images_by_super_class_tensor(coarse_labels_id, fine_labels_id, data_train ):
    numpy_coarse_labels = np.array(data_train[b'coarse_labels'])
    numpy_data = np.array(data_train[b'data'])
    numpy_fine_labels = np.array(data_train[b'fine_labels'])

    indices = np.argwhere(np.where( (numpy_fine_labels == fine_labels_id) * (numpy_coarse_labels == coarse_labels_id) , 1, 0))
    images = numpy_data[indices]
    ds = tf.data.Dataset.from_tensor_slices(images).map(lambda x: x[0] )
    if NORMALIZE_INPUT:
        ds = ds.map(lambda x: tf.cast(x,dtype=tf.float32) / N_255) # normalize pixels 
    return ds

def images_train_to_dataset(data):
    numpy_data_images = np.array(data[b'data'])
    ds = tf.data.Dataset.from_tensor_slices(numpy_data_images)
    if NORMALIZE_INPUT:
        ds = ds.map(lambda x: tf.cast(x,dtype=tf.float32) / N_255) # normalize pixels 
    return ds

def exponential_decay(lr0:float, s:int):
    def exponential_decay_function(epoch):
        return lr0 * 0.1**(epoch/s)
    return exponential_decay_function

def power_scheduling(lr0: float, s:int, c:float):
    def power_scheduling_function(epoch):
        return lr0/(1 + (epoch/s))**c
    return power_scheduling_function

class SaveModelsCallback(tf.keras.callbacks.Callback):
    def __init__(self, decoder, encoder):
        self.decoder = decoder 
        self.encoder = encoder 
        super().__init__()
    def on_epoch_end(self, epoch, logs):
        self.decoder.save(os.path.join(os.getcwd(), 'ex9_dropout_decoder.keras'))
        self.encoder.save(os.path.join(os.getcwd(), 'ex9_dropout_encoder.keras'))

@tf.keras.utils.register_keras_serializable(package="HtronLayers")
class AddDenseBlock(tf.keras.Layer):
    def __init__(self, n_neurons,n_layers,**kwargs):
        super().__init__(**kwargs)
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        self.hidden = [tf.keras.layers.Dense(n_neurons, activation='elu') for _ in range(n_layers) ]
    def call(self, inputs):
        Z = inputs 
        for layer in self.hidden:
            Z = layer(Z)
        return inputs + Z 
    # def get_config(self):
    #     base_config = super().get_config() 
    #     return { **base_config, "hidden": self.hidden, "n_neurons": self.n_neurons, "n_layers": self.n_layers}

train_file = os.path.join(DS_DIR, 'train')
test_file = os.path.join(DS_DIR, 'test')

data_train = unpickle(train_file)
data_test = unpickle(test_file)

# filenames = data[b'filenames']
batch_label = data_train[b'batch_label']
ds_coarse_labels = tf.data.Dataset.from_tensor_slices(data_train[b'coarse_labels'])
ds_fine_labels = tf.data.Dataset.from_tensor_slices(data_train[b'fine_labels'])
ds_images = tf.data.Dataset.from_tensor_slices(data_train[b'data'])

train_fish_fish_ds = images_by_super_class_tensor(1,1, data_train)
test_fish_fish_ds = images_by_super_class_tensor(1,1, data_test)
train_data_ds = images_train_to_dataset(data_train)
test_data_ds = images_train_to_dataset(data_test)

ds_train = tf.data.Dataset.zip( (train_data_ds),(train_data_ds))
ds_train = ds_train.batch(BATCH_SIZE).prefetch(1) 

ds_test = tf.data.Dataset.zip( (test_data_ds),(test_data_ds))
ds_test = ds_test.batch(BATCH_SIZE).prefetch(1) 

minval=-0.1
maxval = 0.1
inner_dense_depth = 2
high_dense_depth = 0

hidden_neurons = [32, 64, 128]
coding_size = hidden_neurons[2] * 4 * 4 

if sys.argv.__len__() <= 1:
    raise ValueError('Are you training or inferring?')
    
if sys.argv[1] == 'train':
    z_in_enc = tf.keras.layers.Input(shape=[32 * 32 * 3])
    z = z_in_enc
    z= tf.keras.layers.Dropout(0.3) (z)
    z = tf.keras.layers.Reshape([32, 32, 3])(z)
    z = tf.keras.layers.Conv2D(hidden_neurons[0], kernel_size = 3, padding='same', activation='selu'   )(z)
    z = tf.keras.layers.MaxPool2D(pool_size=2)(z)
    z = tf.keras.layers.Conv2D(hidden_neurons[1], kernel_size = 3, padding='same', activation='selu'  )(z)
    z = tf.keras.layers.MaxPool2D(pool_size=2)(z)
    z = tf.keras.layers.Conv2D(hidden_neurons[2], kernel_size = 3, padding='same', activation='selu'  )(z)
    z = tf.keras.layers.MaxPool2D(pool_size=2)(z)
    z = tf.keras.layers.Flatten()(z)
    z_out = AddDenseBlock(hidden_neurons[2] * 4 * 4,3)(z)
    dropout_encoder = tf.keras.Model(inputs=[z_in_enc], outputs=[z_out])

    z_in_dec = tf.keras.layers.Input(shape= [4*4*hidden_neurons[2]]) # 32 = 2**5
    z = z_in_dec
    z = tf.keras.layers.Reshape([4, 4, hidden_neurons[2]])(z)
    z = tf.keras.layers.Conv2DTranspose(hidden_neurons[1], kernel_size = 2, strides=2, padding='valid', activation='selu'  )(z)
    z = tf.keras.layers.Conv2DTranspose(hidden_neurons[0], kernel_size = 2, strides=2, padding='same', activation='selu'  )(z)
    z = tf.keras.layers.Conv2DTranspose(3, kernel_size = 2, strides=2, padding='same', activation='sigmoid')(z)
    z_out = tf.keras.layers.Flatten()(z)

    dropout_decoder= tf.keras.Model(inputs=[z_in_dec], outputs=[z_out])
    codings_ = dropout_encoder(z_in_enc)
    reconstruction_ = dropout_decoder(codings_)
    
    model_ae = tf.keras.Model(inputs=[z_in_enc], outputs=[reconstruction_])

    model_ae.compile(loss=tf.keras.losses.MeanAbsoluteError(), optimizer=tf.keras.optimizers.SGD(momentum=0.0012, nesterov=False) ) 

    model_ae.save(os.path.join(os.getcwd(), 'ex9_autoencoder.keras'))


    tf.keras.utils.plot_model(model_ae, to_file=os.path.join(os.getcwd(), 'ex9_autoencoder.png'), expand_nested= True ,show_layer_names=True, show_layer_activations=True, show_trainable=True, show_shapes=True)
    tf.keras.utils.plot_model(dropout_decoder, to_file=os.path.join(os.getcwd(), 'ex9_dropout_decoder.png'), expand_nested= True ,show_layer_names=True, show_layer_activations=True, show_trainable=True, show_shapes=True)
    tf.keras.utils.plot_model(dropout_encoder, to_file=os.path.join(os.getcwd(), 'ex9_dropout_encoder.png'), expand_nested= True ,show_layer_names=True, show_layer_activations=True, show_trainable=True, show_shapes=True)

    history = model_ae.fit( ds_train, epochs=N_EPOCHS, validation_data=(ds_test), callbacks= [
         tf.keras.callbacks.LearningRateScheduler(exponential_decay(0.500, 50))
        #  tf.keras.callbacks.LearningRateScheduler(power_scheduling(0.000000000011211, 2, 1.0))
        # tf.keras.callbacks.LearningRateScheduler(power_scheduling(1.90, 10, 1.0))
         , tf.keras.callbacks.TensorBoard(  os.path.join(os.path.join(os.getcwd(), 'tb_log'), '01' )),  SaveModelsCallback(dropout_decoder,dropout_encoder ) ,tf.keras.callbacks.ModelCheckpoint ( os.path.join(os.getcwd(), 'ex9_autoencoder.keras'), save_best_only=True ), tf.keras.callbacks.BackupAndRestore(os.path.join(os.getcwd(), 'ex9_backup'), save_freq='epoch', delete_checkpoint=True)] )

elif sys.argv[1] == 'infer':
    model_ae = tf.keras.models.load_model(os.path.join(os.getcwd(), 'ex9_autoencoder.keras'))
    
    dropout_decoder_ae = tf.keras.models.load_model(os.path.join(os.getcwd(), 'ex9_dropout_decoder.keras'))

    dropout_encoder_ae = tf.keras.models.load_model(os.path.join(os.getcwd(), 'ex9_dropout_encoder.keras'))


    # ds_test_ = ds_test.map(lambda x: tf.reshape(x, (1, 3072)))

    # # Visualization using autoencoder
    # fig = plt.figure(layout='constrained')
    # index = 0 
    # x_image_compress = dropout_encoder.predict(ds_test_)
    # tsne = TSNE(perplexity = 30)
    # x_test_2D = tsne.fit_transform(x_image_compress)
    # y_test = list(ds_test_coarse_labels.as_numpy_iterator()) 
    # plt.scatter(x_test_2D[:,0], x_test_2D[:,1],  s=1.5, c = y_test,  cmap='tab10')
    
    n_images = 6
    batch_size = 12
    fig3 = plt.figure(3,layout='constrained')
    fig2 = plt.figure(2,layout='constrained')
    fig1 = plt.figure(1,layout='constrained')
    grid_cells = gridspec.GridSpec(n_images, 8, figure=fig1, hspace=0.05, wspace=0.05)
    # ds_test = ds_test.batch(3)

    # test_fish_ds = images_by_super_class_tensor(1, 1, data_train).batch(batch_size).prefetch(1)
    test_fish_ds = images_train_to_dataset(data_train).batch(batch_size).prefetch(1)  
    writer = tf.summary.create_file_writer(os.path.join(os.path.join(os.getcwd(), 'tb_log'), '01_writer' ))
    
    for x_images in test_fish_ds.take(1):

        for batch_i, x_image_serial_norm in enumerate(x_images): 
            
            x_image_serial_norm = tf.cast(x_image_serial_norm, tf.float32)

            # RAW IMAGE
            img = process_serialize_img_data(x_image_serial_norm )
            
            # RECONSTRUCTION IMAGE
            x_image_compress_serial = dropout_encoder_ae.predict( x_image_serial_norm[tf.newaxis,...] )
            x_image_reconstruct_serial = dropout_decoder_ae(x_image_compress_serial)
            x_image_reconstruct_serial = tf.reshape(x_image_reconstruct_serial, (-1))[::]
            print('MAX VALUE ', max(x_image_reconstruct_serial))
            ax3 = fig3.add_subplot(1, 1, 1)
            residual_y = tf.math.abs(x_image_serial_norm - x_image_reconstruct_serial)
            residual_x = tf.range(0, len(residual_y))
            ax3.scatter(residual_x, residual_y)
            ax3.set_title('residual')   

            ax2 = fig2.add_subplot(n_images,1,batch_i + 1)
            ax2.scatter(np.arange(len(x_image_serial_norm)), x_image_serial_norm, c='blue', s=1, label='img')
            ax2.scatter(np.arange(len(x_image_reconstruct_serial)), x_image_reconstruct_serial, c='red', alpha=0.8, s=1, label='img recon')
            ax2.legend()
            img_recon = process_serialize_img_data(x_image_reconstruct_serial)
            
            with writer.as_default():
                tf.summary.image('img1', img[tf.newaxis, ...],1)
                tf.summary.image('img2', img_recon[tf.newaxis, ...],1)


            print(f'img={img}\timg_recon = {img_recon}')
            grid_id_left = gridspec.GridSpecFromSubplotSpec(1,1, subplot_spec=grid_cells[batch_i, 0])
            grid_id_right = gridspec.GridSpecFromSubplotSpec(1,1, subplot_spec=grid_cells[batch_i, 1])

            ax = fig1.add_subplot(grid_id_left[0])
            ax.imshow(img)
            ax.set_axis_off()

            ax = fig1.add_subplot(grid_id_right[0])
            ax.imshow((img_recon))
            ax.set_axis_off()

            if batch_i >=n_images-1:
                break
            
    plt.show()
