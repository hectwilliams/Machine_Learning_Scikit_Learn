'''
    Q. Try using a denoising autoencodder to pretrain an image classifier. 

    A. I will train CIFAR - 100 dataset

    Q. Split dataset into training and test set. Train a deep denoising autoencoder on full training set

    -> A.  Training is on-going. Current loss = 0.0616. My target is 0.03 where the reconstruction of images is good.

    Q. Build a classification DNN, reusing the lower layers of the autoencoder. Train it using only 500 images from the training set. Does it perform better with or without pretraining?

    A.  Pretraining helps the model converge to lower loss 
 
'''
# Download CFAR-100 into current working directory: see https://www.cs.toronto.edu/~kriz/cifar.html 
# unzip cifar-100-python.tar.gz (unzipped directory should be cifar-100-python)
#!/usr/bin/env python3
import os
import numpy as np
import pickle
import tensorflow as tf 
import sys 
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE 
from matplotlib.widgets import Button, Slider
from matplotlib.backend_bases import MouseEvent, Event

DS_DIR = os.path.join('/Users/hectorwilliams/Downloads', 'cifar-100-python')
N_255 = 255.0
N_EPOCHS = 3000
N_EPOCHS_DNN =  3000
BATCH_SIZE = 2
NORMALIZE_INPUT = True 
N_TRAINING_INSTANCES_CFAR100 = 50000
N_INSTANCES_CLASSIFY_APPLICATION = 200
COARSE_LABELS = [ # superclasses 
    'aquatic mammals',
    'fish',
    'flowers',
    'food container',
    'fruit and vegetables',
    'household electrical devices',
    'household furniture',
    'insects',
    'large carnivores',
    'large man-made outdoor things',
    'large natural outdoor scenes',
    'large omnivores and herbivores',
    'medium-sized mammals',
    'non-insect invertbrates',
    'people',
    'reptiles',
    'small mammals',
    'trees',
    'vehicles 1',
    'vehicles 2'
]
N_COARSE_LABELS = len(COARSE_LABELS)

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
    ds = tf.data.Dataset.from_tensor_slices(np.array(data[b'data']))
    if NORMALIZE_INPUT:
        ds = ds.map(lambda x: tf.cast(x,dtype=tf.float32) / N_255) # normalize pixels 
    return ds

def images_train_to_dataset_with_label(data):
    ds = tf.data.Dataset.from_tensor_slices(np.array(data[b'data']))
    labels_ds = tf.data.Dataset.from_tensor_slices(np.array(data[b'coarse_labels']))
    if NORMALIZE_INPUT:
        ds = ds.map(lambda x: tf.cast(x,dtype=tf.float32) / N_255) # normalize pixels 
    return tf.data.Dataset.zip((ds), (labels_ds))

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

def reset_click_handler(event: Event, obj):
    obj[0] = 0
    pass

def button_handler(event: Event, obj, axes_img, axes_target_label, axes_estimate_label, codings, coding_labels, limit, step):
    
    # update shared index value
    current_index = obj[0]

    if current_index == 0 and step == -1:
        current_index = limit
    
    if current_index == limit - 1 and step == 1:
        current_index = -1

    current_index = current_index + step 
    
    obj[0] = current_index
    
    # update axes 
    axes_img.cla()
    axes_target_label.cla()
    axes_estimate_label.cla()

    coding = codings[current_index]
    img = process_serialize_img_data(coding)

    axes_img.imshow(img)
    label_index = coding_labels[current_index]  
    label = COARSE_LABELS[label_index]

    load_sample(axes_img, axes_target_label, axes_estimate_label, coding, label)

    # add update task to gui event loop
    event.canvas.draw_idle()

def load_sample(axes_img, axes_target, axes_estimate, coding, label):
    
    for ax in [axes_img, axes_target, axes_estimate ]:
        ax.set_axis_off()
        ax.set(xticklabels=[])
        ax.set(yticklabels=[])

    img = process_serialize_img_data(coding)
    axes_img.imshow(img)

    axes_target.text(0.5, 0.5, label, horizontalalignment='center', verticalalignment='center', transform=axes_target.transAxes, fontsize=8)

    estimate_label = COARSE_LABELS[tf.math.argmax(dnn_classifier_model.predict(coding[tf.newaxis,...]), axis=1)[0]]    

    match_found = estimate_label == label
    if not match_found:
        # add strikethrough to each text character
        str_buffer = ''
        for c in estimate_label:
            str_buffer += c + '\u0336'
        estimate_label = str_buffer

    axes_estimate.text(0.5, 0.5, estimate_label, horizontalalignment='center', verticalalignment='center', transform=axes_estimate.transAxes, fontsize=8,  bbox=dict(facecolor='none', edgecolor='green' if match_found else 'red', boxstyle='round') )

train_file = os.path.join(DS_DIR, 'train')
test_file = os.path.join(DS_DIR, 'test')

data_train = unpickle(train_file)
data_test = unpickle(test_file)


batch_label = data_train[b'batch_label']
ds_coarse_labels = tf.data.Dataset.from_tensor_slices(data_train[b'coarse_labels'])
ds_fine_labels = tf.data.Dataset.from_tensor_slices(data_train[b'fine_labels'])
ds_images = tf.data.Dataset.from_tensor_slices(data_train[b'data'])

train_fish_fish_ds = images_by_super_class_tensor(1,1, data_train)
test_fish_fish_ds = images_by_super_class_tensor(1,1, data_test)
train_data_ds = images_train_to_dataset(data_train)
test_data_ds = images_train_to_dataset(data_test)

ds_train_autoencoder = tf.data.Dataset.zip( (train_data_ds),(train_data_ds))
ds_train_autoencoder = ds_train_autoencoder.batch(BATCH_SIZE).prefetch(1) 

ds_test = tf.data.Dataset.zip( (test_data_ds),(test_data_ds))
ds_test = ds_test.batch(BATCH_SIZE).prefetch(1) 

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

    history = model_ae.fit( ds_train_autoencoder, epochs=N_EPOCHS, validation_data=(ds_test), callbacks= [
         tf.keras.callbacks.LearningRateScheduler(exponential_decay(0.500, 50))
        # tf.keras.callbacks.LearningRateScheduler(power_scheduling(1.90, 10, 1.0))
         , tf.keras.callbacks.TensorBoard(  os.path.join(os.path.join(os.getcwd(), 'tb_log'), '01' )),  SaveModelsCallback(dropout_decoder,dropout_encoder ) ,tf.keras.callbacks.ModelCheckpoint ( os.path.join(os.getcwd(), 'ex9_autoencoder.keras'), save_best_only=True ), tf.keras.callbacks.BackupAndRestore(os.path.join(os.getcwd(), 'ex9_backup'), save_freq='epoch', delete_checkpoint=True)] )

elif sys.argv[1] == 'infer':
    model_ae = tf.keras.models.load_model(os.path.join(os.getcwd(), 'ex9_autoencoder.keras'))
    
    dropout_decoder_ae = tf.keras.models.load_model(os.path.join(os.getcwd(), 'ex9_dropout_decoder.keras'))

    dropout_encoder_ae = tf.keras.models.load_model(os.path.join(os.getcwd(), 'ex9_dropout_encoder.keras'))
    
    n_images = 6
    batch_size = 12
    fig3 = plt.figure(3,layout='constrained')
    fig2 = plt.figure(2,layout='constrained')
    fig1 = plt.figure(1,layout='constrained')
    grid_cells = gridspec.GridSpec(n_images, 8, figure=fig1, hspace=0.05, wspace=0.05)
    test_fish_ds = images_train_to_dataset(data_test).batch(batch_size).prefetch(1)   
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

            ax2 = fig2.add_subplot(n_images,1,batch_i + 1)
            ax2.scatter(np.arange(len(x_image_serial_norm)), x_image_serial_norm, c='blue', s=1, label='img')
            ax2.scatter(np.arange(len(x_image_reconstruct_serial)), x_image_reconstruct_serial, c='red', alpha=0.8, s=1, label='img recon')
            ax2.legend()
            img_recon = process_serialize_img_data(x_image_reconstruct_serial)

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

elif sys.argv[1] == 'dnn-train':
    
    encoder_model = tf.keras.models.load_model(os.path.join(os.getcwd(), 'ex9_dropout_encoder.keras'))
    
    # for layer in encoder_model.layers:
        # layer.trainable =  False 
    encoder_model.compile()
    
    z_input_dnn = tf.keras.layers.Input(shape=[32 * 32 * 3])
    z  = encoder_model(z_input_dnn)
    z_output_dnn = tf.keras.layers.Dense(N_COARSE_LABELS * 2, activation='relu')(z)
    z_output_dnn = tf.keras.layers.Dense(N_COARSE_LABELS, activation='softmax')(z)

    dnn_model = tf.keras.Model(inputs=[z_input_dnn], outputs=[z_output_dnn])
    dnn_model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd')    

    ds_train_dnn = images_train_to_dataset_with_label(data_train).shuffle(N_TRAINING_INSTANCES_CFAR100).take(500).batch(1)
    ds_test_dnn = images_train_to_dataset_with_label(data_test).take(1500).batch(1)
    
    dnn_model.fit(ds_train_dnn, validation_data=[ds_test_dnn], epochs=N_EPOCHS_DNN, callbacks=[tf.keras.callbacks.LearningRateScheduler(exponential_decay(0.00100, 100)), tf.keras.callbacks.ModelCheckpoint ( os.path.join(os.getcwd(), 'ex9_DNNClassifierPretrained.keras'), save_best_only=True ), tf.keras.callbacks.BackupAndRestore(os.path.join(os.getcwd(), 'ex9_backup_dnn_enc'), save_freq='epoch', delete_checkpoint=True)] )

elif sys.argv[1] == 'dnn-infer':

    # dnn_classifier_model = tf.keras.models.load_model(os.path.join(os.getcwd(), 'ex9_DNNClassifier.keras'))
    dnn_classifier_model = tf.keras.models.load_model(os.path.join(os.getcwd(), 'ex9_DNNClassifierPretrained.keras'))
    fig2_grid_layout = plt.figure(2,layout='tight')
    grid_columns = 16
    grid_rows = 16
    grid_spec = gridspec.GridSpec(grid_rows, grid_columns, figure=fig2_grid_layout,  hspace=0.05, wspace=0.05)
    
    for row in range(grid_rows):
        for col in range(1,grid_columns) :
            ax = fig2_grid_layout.add_subplot(grid_spec[row, col])
            ax.set_axis_off()
            ax.set(xticklabels=[])
            ax.set(yticklabels=[])
            ax.tick_params(bottom=False, top=False, left=False, right=False)
         
    # button 1 
    button_next_subplot_spec = gridspec.GridSpecFromSubplotSpec(1,1, subplot_spec=grid_spec[-2, 5:8])
    ax = fig2_grid_layout.add_subplot(button_next_subplot_spec[0,0])
    button_back = Button(ax, 'Back', hovercolor='0.975', color='red')

    # button 2
    button_reset_subplot_spec = gridspec.GridSpecFromSubplotSpec(1,1, subplot_spec=grid_spec[-2, 9:12])
    ax = fig2_grid_layout.add_subplot(button_reset_subplot_spec[0,0])
    button_next = Button(ax, 'Next ', hovercolor='0.975', color='red')

    # images section 
    images_subplot_spec = gridspec.GridSpecFromSubplotSpec(1,1, subplot_spec=grid_spec[3:7, :])
    ax_img = fig2_grid_layout.add_subplot(images_subplot_spec[0,0])

    # text target section
    target_label_subplot_spec = gridspec.GridSpecFromSubplotSpec(1,1, subplot_spec=grid_spec[8:9, :])
    ax_ttarget = fig2_grid_layout.add_subplot(target_label_subplot_spec[0,0])

    # text estimated target section 
    estimate_label_subplot_spec = gridspec.GridSpecFromSubplotSpec(1,1, subplot_spec=grid_spec[9:10, :])
    ax_testimated = fig2_grid_layout.add_subplot(estimate_label_subplot_spec[0,0])

    index_obj = [0] 
    ds_test = images_train_to_dataset_with_label(data_test).take(N_INSTANCES_CLASSIFY_APPLICATION).batch(N_INSTANCES_CLASSIFY_APPLICATION, drop_remainder=True).prefetch(1)
    x_images_codings_ = None  
    x_labels_ = None
    
    # load/parse dataset
    for x_images, x_label in ds_test.take(1):
        x_images_codings_ = x_images 
        x_labels_ = x_label 

    button_back.on_clicked(lambda event: button_handler(event, index_obj, ax_img, ax_ttarget, ax_testimated, x_images_codings_, x_labels_, N_INSTANCES_CLASSIFY_APPLICATION, -1))
    button_next.on_clicked(lambda event: button_handler(event, index_obj, ax_img, ax_ttarget, ax_testimated, x_images_codings_, x_labels_, N_INSTANCES_CLASSIFY_APPLICATION, 1))
    
    #initialize clasifier application 
    load_sample(ax_img,ax_ttarget,ax_testimated, x_images_codings_[index_obj[0]], COARSE_LABELS[x_labels_[0]]  )

    plt.show()
