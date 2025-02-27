#!/usr/bin/env python3
import os
import numpy as np
import pickle
import tensorflow as tf 
import sys 
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE 

DS_DIR = os.path.join('/Users/hectorwilliams/Downloads', 'cifar-100-python')
N_255 = 255.0
N_EPOCHS = 60
BATCH_SIZE = 2
NORMALIZE_INPUT = True 

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def process_serialize_img_data(serialize_data_norm):
    xplit = np.array_split( tf.cast(serialize_data_norm  * (N_255 if  NORMALIZE_INPUT else 1) , tf.uint8), 3)
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
        ds = ds.map(lambda x: tf.cast(x,dtype=tf.float16) / N_255) # normalize pixels 
    return ds

train_file = os.path.join(DS_DIR, 'train')
test_file = os.path.join(DS_DIR, 'test')

data_train = unpickle(train_file)
data_test = unpickle(test_file)

# TRAIN SET
# filenames = data[b'filenames']
batch_label = data_train[b'batch_label']
ds_coarse_labels = tf.data.Dataset.from_tensor_slices(data_train[b'coarse_labels'])
ds_fine_labels = tf.data.Dataset.from_tensor_slices(data_train[b'fine_labels'])
ds_images = tf.data.Dataset.from_tensor_slices(data_train[b'data'])

train_fish_fish_ds = images_by_super_class_tensor(1,1, data_train)

if NORMALIZE_INPUT:
    ds_images = ds_images.map(lambda x: tf.cast(x,dtype=tf.float16) / N_255) # normalize pixels 
ds_train = tf.data.Dataset.zip( (train_fish_fish_ds),(train_fish_fish_ds))
ds_train = ds_train.batch(BATCH_SIZE).prefetch(1) 

# TEST SET
batch_label = data_test[b'batch_label']
ds_coarse_labels = tf.data.Dataset.from_tensor_slices(data_test[b'coarse_labels'])
ds_fine_labels = tf.data.Dataset.from_tensor_slices(data_test[b'fine_labels'])
ds_images = tf.data.Dataset.from_tensor_slices(data_test[b'data'])

if NORMALIZE_INPUT:
    ds_images = ds_images.map(lambda x: tf.cast(x,dtype=tf.float16) / N_255) # normalize pixels 
# ds_test = tf.data.Dataset.zip( (ds_images) ).batch(32).prefetch(1)
ds_valid, ds_test = tf.keras.utils.split_dataset(ds_images, left_size=0.60)
ds_valid_ = tf.data.Dataset.zip( (ds_valid),(ds_valid)).batch(BATCH_SIZE).prefetch(1) 
ds_valid_coarse_labels ,ds_test_coarse_labels = tf.keras.utils.split_dataset(ds_fine_labels, left_size=0.60)
ds_valid_fine_labels ,ds_test_fine_labels = tf.keras.utils.split_dataset(ds_fine_labels, left_size=0.60)

coding_size = 650 

minval=-0.1
maxval = 0.1
inner_dense_depth = 20
high_dense_depth = 20
z_in_enc = tf.keras.layers.Input(shape=[32 * 32 * 3])

z = tf.keras.layers.Dense(1000, activation='selu')(z_in_enc)

for neurons in [1000]:
    for _ in range(high_dense_depth):
        z = tf.keras.layers.Dense(neurons, activation='selu')(z)

for neurons in [128, 256, 512]:
    for _ in range(inner_dense_depth):
        z = tf.keras.layers.Dense(neurons, activation='selu')(z)

z = tf.keras.layers.Dense(64, activation='selu')(z)

z_out = tf.keras.layers.Dense(coding_size, activation='selu')(z)

dropout_encoder = tf.keras.Model(inputs=[z_in_enc], outputs=[z_out])


z_in_dec = tf.keras.layers.Input(shape= [coding_size])

z = tf.keras.layers.Dense(64, activation='selu')(z_in_dec)

for neurons in [128, 256, 512][::-1]:
    for _ in range(inner_dense_depth):
        z = tf.keras.layers.Dense(neurons, activation='selu')(z)

for neurons in [1000]:
    for _ in range(high_dense_depth):
        z = tf.keras.layers.Dense(neurons, activation='selu')(z)

z = tf.keras.layers.Dense(1000, activation='selu')(z)

z_out = tf.keras.layers.Dense(32 * 32 * 3, activation='sigmoid')(z)

dropout_decoder= tf.keras.Model(inputs=[z_in_dec], outputs=[z_out])
codings_ = dropout_encoder(z_in_enc)
reconstruction_ = dropout_decoder(codings_)

if sys.argv.__len__() <= 1:
    raise ValueError('Are you training or inferring?')
    
if sys.argv[1] == 'train':

    # model_ae = tf.keras.Sequential([dropout_encoder, dropout_decoder])
    
    model_ae = tf.keras.Model(inputs=[z_in_enc], outputs=[reconstruction_])

    model_ae.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(decay=1e-4, learning_rate=0.001, momentum=0.4, nesterov=False) )
    # categorical_crossentropy

    history = model_ae.fit( ds_train, epochs=N_EPOCHS, validation_data=(ds_train), callbacks= [ tf.keras.callbacks.TensorBoard(  os.path.join(os.path.join(os.getcwd(), 'tb_log'), '01' )), tf.keras.callbacks.ModelCheckpoint ( os.path.join(os.getcwd(), 'ex9_autoencoder.keras'), save_best_only=True ), tf.keras.callbacks.BackupAndRestore(os.path.join(os.getcwd(), 'ex9_backup'), save_freq='epoch', delete_checkpoint=True)] )

    model_ae.save(os.path.join(os.getcwd(), 'ex9_autoencoder.keras'))

    tf.keras.utils.plot_model(model_ae, to_file=os.path.join(os.getcwd(), 'ex9_autoencoder.png'), expand_nested= True ,show_layer_names=True, show_layer_activations=True, show_trainable=True, show_shapes=True)

elif sys.argv[1] == 'infer':
    model_ae = tf.keras.models.load_model(os.path.join(os.getcwd(), 'ex9_autoencoder.keras'))
    ds_test_ = ds_test.map(lambda x: tf.reshape(x, (1, 3072)))

    # # Visualization using autoencoder
    # fig = plt.figure(layout='constrained')
    # index = 0 
    # x_image_compress = dropout_encoder.predict(ds_test_)
    # tsne = TSNE(perplexity = 30)
    # x_test_2D = tsne.fit_transform(x_image_compress)
    # y_test = list(ds_test_coarse_labels.as_numpy_iterator()) 
    # plt.scatter(x_test_2D[:,0], x_test_2D[:,1],  s=1.5, c = y_test,  cmap='tab10')
    
    n_images=2
    fig3 = plt.figure(3,layout='constrained')
    fig2 = plt.figure(2,layout='constrained')
    fig1 = plt.figure(1,layout='constrained')
    grid_cells = gridspec.GridSpec(n_images, 8, figure=fig1, hspace=0.05, wspace=0.05)
    ds_test = ds_test.batch(BATCH_SIZE)

    test_fish_ds = images_by_super_class_tensor(1, 1, data_train).batch(32).prefetch(1)
    writer = tf.summary.create_file_writer(os.path.join(os.path.join(os.getcwd(), 'tb_log'), '01_writer' ))
    for x_images in test_fish_ds.take(1):

        for batch_i, x_image_serial_norm in enumerate(x_images): 
            
            x_image_serial_norm = tf.cast(x_image_serial_norm, tf.float32)

            # RAW IMAGE
            img = process_serialize_img_data(x_image_serial_norm )
            
            # RECONSTRUCTION IMAGE
            x_image_compress_serial = dropout_encoder.predict( x_image_serial_norm[tf.newaxis,...] )
            x_image_reconstruct_serial = dropout_decoder(x_image_compress_serial)
            x_image_reconstruct_serial = tf.reshape(x_image_reconstruct_serial, (-1))
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


            print(f'img={img}\timg_recon = {tf.round(img_recon, 2) }')
            grid_id_left = gridspec.GridSpecFromSubplotSpec(1,1, subplot_spec=grid_cells[batch_i, 0])
            grid_id_right = gridspec.GridSpecFromSubplotSpec(1,1, subplot_spec=grid_cells[batch_i, 1])

            ax = fig1.add_subplot(grid_id_left[0])
            ax.imshow( tf.reshape(img, shape=(32,32,3)))
            ax.set_axis_off()

            ax = fig1.add_subplot(grid_id_right[0])
            ax.imshow(   tf.reshape(img_recon, shape=(32,32,3)))  
            ax.set_axis_off()

            if batch_i >=n_images-1:
                break

    plt.show()
