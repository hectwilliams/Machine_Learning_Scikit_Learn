
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

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

              tf.multiply(self.kernel[channel_id], image [ : , field_record[0] : field_record[1] , field_record[2] : field_record[3]] )

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

# TRAINING DATA
dataset = tfds.load(name='mnist')

mnist_train, mnist_test = dataset['train'], dataset['test'] # dataset houses list of dicts

mnist_train = mnist_train.map(lambda record: (record['image'], record['label'] ))

mnist_test = mnist_test.map(lambda record: (record['image'], record['label'] ))

mnist_val = mnist_train.skip(50000).cache().batch(32, drop_remainder= True)

mnist_train = mnist_train.take(100).cache().batch(32, drop_remainder= True).prefetch(tf.data.experimental.AUTOTUNE)

# mnist_test =  mnist_test.batch(32, drop_remainder= True)

# BUILD MODEL
z = ii = tf.keras.layers.Input(shape=(28,28,1,))

z = tf.keras.layers.Lambda(lambda x: tf.cast(x/255.0, tf.float32))(z)

z = customCNN(layers=64, kernel_w=3, kernel_h=3, stride_w=1, stride_h=1, padding_type='SAME', activation='relu', channel_first=False)(z)

z = tf.keras.layers.MaxPool2D(pool_size=(2,2))(z)

z = tf.keras.layers.Flatten()(z)

oo = tf.keras.layers.Dense(10, activation=tf.keras.layers.Softmax())(z)

model = tf.keras.Model(inputs=[ii], outputs=[oo])


model.compile (loss= tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.SGD(learning_rate=0.0034) )

# TRAIN

model.fit(mnist_train, epochs=10)

# EVALUATE

# model.evaluate(mnist_test)

