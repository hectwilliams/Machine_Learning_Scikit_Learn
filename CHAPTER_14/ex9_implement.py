
'''

  Implement a custom convolutional layer 

  Note: 
  
    - The solution is not optimal!

    - Training, validation, and test batch size must be equal ( tf-function auto-graph requires a static-generic interface )

'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import time 

class customCNN(tf.keras.Layer):

  def __init__(self, layers=1, kernel_w=1, kernel_h=1, stride_w=1, stride_h=1, padding_type = 'SAME', activation = None, channel_first = True, **kwargs):

    super().__init__(**kwargs)

    self.layers_n = layers

    self.kernel_w = kernel_w

    self.kernel_h = kernel_h

    self.stride_w = stride_w

    self.stride_h = stride_h

    self.padding_type = padding_type

    self.channel_first = channel_first

    self.activation = tf.keras.activations.get(activation)

    self.pad_left = tf.Variable(0)

    self.pad_right = tf.Variable(0)

    self.input_batch_size = tf.Variable(1)

    self.batch_set = False

    self.imageMap = {}


  def pad_helper(self, p):
    '''

      pads call methods X argument if feature map requires zero padding

    '''

    self.pad_left.assign(0)

    self.pad_right.assign(0)

    k = tf.cast(p % 2 == 1, tf.int32)

    self.pad_left.assign( tf.cast( (p - k) / 2, tf.int32 ) )

    self.pad_right.assign(self.pad_left + k)

    return self.pad_left, self.pad_right   # top, down or left, right


  def build(self, batch_input_shape):

    '''

      Inputs follow channel first format: batch_size x input_channels x input_height x input_width

    '''
    self.input_width = batch_input_shape[3]

    self.input_height = batch_input_shape[2]

    self.input_channels_n = batch_input_shape[1]

    self.input_batch_size.assign( batch_input_shape[0] or self.input_batch_size )

    if not self.channel_first:

      self.input_channels_n = batch_input_shape[3]

      self.input_height = batch_input_shape[1]

      self.input_width = batch_input_shape[2]

    self.range = tf.range(self.input_batch_size)

    if self.padding_type == 'SAME':

      self.output_width = tf.cast(tf.math.ceil(self.input_width / self.stride_w), tf.dtypes.int32)

      self.output_height = tf.cast(tf.math.ceil(self.input_height / self.stride_h), tf.dtypes.int32)

      self.max_input_w_eff_length = (self.output_width - 1) * self.stride_w + (self.kernel_w - 1) + 1

      self.max_input_h_eff_length = (self.output_height - 1) * self.stride_h + (self.kernel_h - 1) + 1

      self.padding_w = self.max_input_w_eff_length - self.input_width

      self.padding_h =  self.max_input_h_eff_length - self.input_height

    elif self.padding_type == 'VALID':

      self.output_width = tf.cast((self.input_width - self.kernel_w) / self.stride_w, tf.int32) + 1

      self.output_height = tf.cast((self.input_height - self.kernel_h) / self.stride_h, tf.int32) + 1

    else:

      raise ValueError(f'Padding type {self.padding_type} not supported. Use SAME or VALID')

    self.bias = self.add_weight(name='bias', shape=(self.layers_n,), initializer='zeros')

    if self.channel_first:

      self.kernel = self.add_weight(name='kernel', shape=(self.layers_n, self.input_channels_n, self.kernel_h, self.kernel_w), initializer='glorot_normal')

    else:

      self.kernel = self.add_weight(name='kernel', shape=(self.layers_n, self.kernel_h, self.kernel_w, self.input_channels_n), initializer='glorot_normal')

    super().build(batch_input_shape)

  def call(self, X):

    if self.batch_set == False :

      self.batch_set = True

      self.input_batch_size.assign( tf.shape(X)[0] )

      self.output_ = tf.Variable(tf.zeros((self.input_batch_size , self.layers_n, self.output_height, self.output_width)))

      self.serials = tf.Variable(  tf.map_fn(lambda some_img : tf.io.serialize_tensor(some_img) , tf.random.normal(shape=tf.shape(X) ), fn_output_signature=tf.string) , dtype=tf.string)

    if self.padding_type == 'SAME':

      if self.channel_first:

        p = tf.stack((tf.constant([0, 0]), tf.constant([0, 0]), self.pad_helper(self.padding_h), self.pad_helper(self.padding_w)), axis=0)

      else :

        p = tf.stack((tf.constant([0, 0]), self.pad_helper(self.padding_h), self.pad_helper(self.padding_w) ,tf.constant([0, 0]) ), axis=0)

      X = tf.pad(X, p) # last two axis are padding (if needed)

    elif self.padding_type == 'VALID':

      pass

    # Store all output positions (i,j)

    range_h = tf.range(self.output_height)

    repeat_ = tf.repeat(self.output_height, self.output_height)

    b = tf.repeat(range_h, repeat_)[None, ... ]

    c = tf.tile(range_h, [self.output_height])[None, ... ]

    self.ij  = tf.transpose(tf.concat((b,c), axis=0))

    self.ij = tf.cast(self.ij, tf.int32)

    # Possible kernel subfields windows (ranges)

    h_starts = tf.map_fn(lambda ele : ele[0] * self.stride_h ,  self.ij )
    h_ends = h_starts +  self.kernel_h 

    w_starts = tf.map_fn(lambda ele : ele[1] * self.stride_w,  self.ij )
    w_ends = w_starts +  self.kernel_w 

    h = tf.transpose(tf.concat((h_starts[None, ... ], h_ends[None, ... ]), axis=0))
    w = tf.transpose(tf.concat((w_starts[None, ... ], w_ends[None, ... ]), axis=0))

    self.fields = tf.concat((h,w), axis = 1)
    self.fields = tf.cast(self.fields, tf.int32)

    # Hash Table Update

    codes = tf.map_fn(lambda img: tf.io.serialize_tensor(img), X, fn_output_signature=tf.string)
    
    self.serials.assign(codes)

    tf.map_fn( self.convolution_process , X)

    # relu activation 
    return self.activation( self.output_ )
  
  @tf.function
  def block_reduce (self, data_block, neuron_pos_i, neuron_pos_j, ch_id, batch_id):
    '''

      Calculates feature map's neuron impulse response

      Args:

        data_block - neuron subfield block ( single layer block )

        neuron_pos_i - feature map row position

        neuron_pos_j - feature map col position

        ch_id - feature map identifier

        batch_id - batch identifier

      Return:

        neuron impulse response ( amplitude )

    '''

    block_reduce_sum = tf.reduce_sum(data_block)

    self.output_.scatter_nd_update(indices=[[batch_id, ch_id, neuron_pos_i, neuron_pos_j]], updates=[block_reduce_sum] )

    return tf.cast(0.0, tf.float32)

  def update_channel_neurons(self, sfield_block, ch_id, batch_id):
    '''
    
      Calculates feature map's neuron impulse response

      Args:

        sfield_block - neuron subfield block ( single layer block )

        ch_id - feature map identifier

        batch_id - batch identifier

      Return:

        neuron impulse response ( amplitude )
    '''

    # set feature map(channel) neurons
    tf.map_fn(lambda ij: self.block_reduce(sfield_block, ij[0], ij[1], ch_id, batch_id) , self.ij, dtype=tf.float32, fn_output_signature=tf.float32)

    # add bias to channel
    self.output_[batch_id, ch_id].assign( self.output_[batch_id, ch_id] + self.bias[ch_id] )

    return tf.cast(1.0, tf.float32)

  def convolution_process(self, image ):

    '''
      Entry method for convolutional layer. 

      Args:

        image - input image

      Return:

        Dummy variable 
    '''
    
    serial_data = tf.io.serialize_tensor(image)
    
    equal = (tf.equal(self.serials, serial_data))

    img_id = tf.cast(tf.reduce_sum(tf.where(equal)) , tf.int32)

    conv_channel_ids = tf.range(self.layers_n) 

    tf.map_fn(

      # for each channel id

      lambda channel_id:

        # for each sub_field ( convolutional depth-wise sweep)

       self.update_channel_neurons( 
        
        # method sets feature map neurons 'impulse' response

           tf.map_fn(

            lambda field_record:

            (
              # kernel mult input sub-block

              tf.multiply(self.kernel[channel_id], image [ : , field_record[0] : field_record[1]  , field_record[2] : field_record[3]] )

              if self.channel_first else

              tf.multiply(self.kernel[channel_id], image[  field_record[0] : field_record[1] , field_record[2] : field_record[3] ,: ])

            )
           ,

          self.fields,  fn_output_signature =tf.float32) ,channel_id, img_id) ,

        conv_channel_ids,  fn_output_signature =tf.float32)

    return tf.cast(1.0,dtype=tf.float32)

  def compute_output_shape(self, batch_input_shape):

    return tf.TensorShape((batch_input_shape[0], self.layers_n, self.output_height, self.output_width))

  def get_config(self):

    base_config = super().get_config()

    return {**base_config,
            'layers': self.layers_n,
            'kernel_w': self.kernel_w,
            'kernel_h': self.kernel_h,
            'stride_w': self.stride_w,
            'stride_h': self.stride_h,
            'padding_type': self.padding_type,
            'channel_first': self.channel_first,
            'activation': tf.keras.activations.serialize(self.activation)}

BATCH_SIZE = 8

# # TRAINING DATA

dataset = tfds.load(name='mnist')

mnist_train, mnist_test = dataset['train'], dataset['test'] # dataset houses list of dicts

mnist_train = mnist_train.map(lambda record: (record['image'], record['label'] ))

mnist_test = mnist_test.map(lambda record: (record['image'], record['label'] ))

mnist_val = mnist_train.skip(50000) #.cache().batch(32, drop_remainder= True)

mnist_train = mnist_train.take(50000) #.cache().batch(32, drop_remainder= True).prefetch(tf.data.experimental.AUTOTUNE)

mnist_test =  mnist_test.batch(BATCH_SIZE, drop_remainder= True)

mnist_train = mnist_train.cache().batch(BATCH_SIZE, drop_remainder= True)

mnist_val = mnist_val.take(8000).batch(BATCH_SIZE, drop_remainder= True) 

# # BUILD MODEL

z = ii = tf.keras.layers.Input(shape=(28,28,1,))

z = tf.keras.layers.Lambda(lambda x: tf.cast(x/255.0, tf.float32))(z)

z = customCNN(layers=64, kernel_w=3, kernel_h=3, stride_w=1, stride_h=1, padding_type='SAME', activation='relu', channel_first=False)(z)

z = tf.keras.layers.MaxPool2D(pool_size=(2,2))(z)

z = tf.keras.layers.Flatten()(z)

oo = tf.keras.layers.Dense(10, activation=tf.keras.layers.Softmax())(z)

model = tf.keras.Model(inputs=[ii], outputs=[oo])


@tf.function
def custom_training(n_epochs, mnist_train, n_steps_train, optimizer, loss_fn, mean_train_loss, mean_train_accuracy, mnist_val , n_steps_val, mean_val_loss, mean_val_accuracy, train_loss_history, val_loss_history, train_accuracy_history, val_accuracy_history):

  '''

    Custom training loop , with validation. Compiling and fit methods are not used; those processes are replaced by this function.

    Args:

    n_epochs - number of epochs 

    mnist_train - batched training set 

    n_step_train - number of batched elements in mnist_train variable 

    optimizer - Nadam search algorithm

    loss_fn - sparse categorical crossentropy loss 

    mean_train_loss - mean loss variable 

    mean_train_accuracy - training set sparse categorical accuracy 

    mnist_val - batched validation set 

    n_steps_val - number of batched elements in mnist_val variable 

    mean_val_loss - mean validation loss  

    mean_val_accuracy - validation set sparse categorical accuracy 
    
    train_loss_history - training loss history 

    val_loss_history - validation loss history 

    train_accuracy_history - training accuracy history 

    val_accuracy_history - validation accuracy history


  '''
  for epoch in tf.range(n_epochs):
    
    # training set
    for step in tf.range(n_steps_train):

      X_batch, y_batch = next(iter(mnist_train))  
      
      with tf.GradientTape() as tape:
        
        y_pred = model(X_batch, training =True)  # predict (returns tensor tuple)
        
        main_training_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))

        gradients = tape.gradient(main_training_loss, model.trainable_variables)

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
      mean_train_loss(main_training_loss)
        
      mean_train_accuracy.update_state(y_batch, y_pred)
    
    # validation set

    for step  in tf.range(n_steps_val) :

      X_val_batch, y_val_batch = next(iter(mnist_val)) 

      y_pred_val = model( X_val_batch, training=False )   # predict (returns tensor tuple)

      val_training_loss = tf.reduce_mean(loss_fn(y_val_batch, y_pred_val ))

      mean_val_loss(val_training_loss)

      mean_val_accuracy.update_state(y_val_batch, y_pred_val)

    # store avg loss/metrics per epoch 

    train_loss_history.scatter_nd_update(indices=[[epoch]], updates=[mean_train_loss.result()])

    val_loss_history.scatter_nd_update(indices=[[epoch]], updates=[mean_val_loss.result()])

    train_accuracy_history.scatter_nd_update(indices=[[epoch]], updates=[mean_train_accuracy.result()])

    val_accuracy_history.scatter_nd_update(indices=[[epoch]], updates=[mean_val_accuracy.result()])

    # clear loss/metrics variables 

    mean_train_loss.reset_state()

    mean_val_loss.reset_state()

    mean_train_accuracy.reset_state()

    mean_val_accuracy.reset_state()

n_epochs = 1

n_steps = 1

n_steps_train = len(mnist_train) 

optimizer = tf.keras.optimizers.Nadam(learning_rate=0.0034)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

loss_val_fn = tf.keras.losses.SparseCategoricalCrossentropy()

mean_train_loss = tf.keras.metrics.Mean()

mean_train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy() 

n_steps_val = len(mnist_val)

mean_val_loss = tf.keras.metrics.Mean()

mean_val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy() 

train_loss_history = tf.Variable(tf.zeros(n_epochs), dtype=tf.float32)
 
val_loss_history = tf.Variable(tf.zeros(n_epochs), dtype=tf.float32)

train_accuracy_history = tf.Variable(tf.zeros(n_epochs), dtype=tf.float32)
 
val_accuracy_history = tf.Variable(tf.zeros(n_epochs), dtype=tf.float32)

custom_training(n_epochs,mnist_train, n_steps_train, optimizer, loss_fn, mean_train_loss, mean_train_accuracy, mnist_val , n_steps_val, mean_val_loss, mean_val_accuracy, train_loss_history, val_loss_history, train_accuracy_history, val_accuracy_history) 


# PRINT STATS 

print( f'  Avg training loss =  {tf.reduce_mean(train_loss_history)}')
print( f'  Avg training accuracy =  {(train_accuracy_history)}')

print( f'  Avg validation loss =  {tf.reduce_mean(val_loss_history)}')
print( f'  Avg validation accuracy =  {(val_accuracy_history)}')

