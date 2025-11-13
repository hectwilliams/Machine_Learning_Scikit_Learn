'''

  Model uses Concatenative Attention to create date to iso-date translator

  1) Set DEBUG = False

  2) Set INFER = False 

  3) Run Model  ( - .weights.h5 file generated)

  4) Set INFER = True

  5) Run Model  ( Model will translate "September 01, 1987" into 1987-11-01  )

  Note: 
  
  -- Model translation limited to range [November 14 1955 , November 11 2025]

  -- Select between GRU or LSTM architecture using constant variable USE_GRU
  
  Enjoy :)

'''
# %load_ext tensorboard
# %tensorboard --logdir logs/fit

import os

import re

import numpy as np

import tensorflow as tf

import keras 

from datetime import datetime, timedelta

from collections import Counter

USE_GRU = True

DROPOUT_FACTOR = 0.8

DROPOUT_EN = False

DEBUG = False

INFER = False

N_EMBEDDING  = 128

NUM_OOV_BUCKETS = 1

N_DATES = 70

MAX__TIME_SAMPLES = 6

N_EPOCHS = 15

NEURONS_LSTM = 128

ENCODER_NEURONS_LSTM = NEURONS_LSTM

LEARNING_RATE = 0.0324

LEARNING_DECAY_STEPS = 10000

LEARNING_DECAY_RATE = 0.94

LOG_DIR = 'logs/fit/' + datetime.now().strftime("%Y%m%d-%H:%M:%S")

WEIGHTS_FILENAME = 'enc_dec.weights.h5'

MODEL_FILENAME = 'enc_dec.keras'

MODEL_CHECKPOINT = 'checkpoint_path.ckpt"'

MODEL_NAME = 'model.keras'

USE_BIDIRECTIONAL = True

USE_ATTENTION = True

BEAM_SEARCH_K = 3

FACTOR = 8

BATCH_SIZE = 8

NEW_ATTENTION = True

DECODE_EOS_ID = 6

def to_iso_8601(date_string):

  date_string = date_string.lower()

  ret = re.match(r'([a-z]+)\s*,\s*(\d+)\s*(\d+)', date_string)

  g = ret.groups()

  month = (g[0])

  day = (g[1])

  year = (g[2])

  m = {

    'january': '01',
    'february': '02',
    'march': '03',
    'april': '04',
    'may': '05',
    'june': '06',
    'july': '07',
    'august': '08',
    'september': '09',
    'october': '10',
    'november': '11',
    'december': '12'

  }[month]

  return f'{year}-{m}-{day} '

def dates(n_years):

  end_date = datetime.now()

  start_date = end_date - timedelta(days=n_years*365.25) # Account for leap years

  dates = []

  current_date = start_date

  while current_date <= end_date:

    dates.append(current_date.strftime("%B, %d %Y"))

    current_date += timedelta(days=1)

  return np.array(dates)

def test_vector(string, lookuptable, max_time_samples, reverse=False):
  """
    splits, encodes, and zero - pads (header) raw date entry
  """

  e = string.split()

  e = tf.constant(e)

  test_vector_ = lookuptable.lookup(e) [..., None]

  n_pad = max_time_samples - tf.shape(test_vector_)[0]

  n_pad_data = (tf.repeat([0] , [n_pad])[...,None])

  n_pad_data = tf.cast(n_pad_data, tf.int64)

  test_vector_ = tf.concat ( (  test_vector_, n_pad_data  ),0)

  return test_vector_

def pad_vector(vector, max_time_samples):

  n_pad = max_time_samples - tf.shape(vector)[0]

  n_pad_data = (tf.repeat([0] , [n_pad])[...,None])

  n_pad_data = tf.cast(n_pad_data, tf.int64)

  vector = tf.concat ( (  vector, n_pad_data  ),0)

  return vector

def inference(words, encoder_vocab, decoder_table, encoder_vector, model, encoder_model, decoder_model ):

  """

    word_i -> [encode] -> model() -> predict_a_word_i

  """

  enc_ret = encoder_model( x = encoder_vector[None, ...] )

  e, h, c, h2, c2 = enc_ret;

  decode_vector  = decoder_table.lookup(tf.constant(words)) [..., None]

  decode_vector = pad_vector(decode_vector, MAX__TIME_SAMPLES)

  decode_vector = tf.cast(decode_vector, tf.int64)

  probabilities = decoder_model(enc_ret)

  probabilities = tf.squeeze(probabilities,axis=0)

  probabilities_current_time_step = probabilities[len(words) - 1]

  return probabilities_current_time_step

def gen_lookup(vocab:list):

  words = tf.constant(vocab)

  words = tf.unique(words).y

  words_ids = tf.range( tf.shape(words)[-1], dtype=tf.int64)

  vocab_init = tf.lookup.KeyValueTensorInitializer(words, words_ids)

  table = tf.lookup.StaticVocabularyTable(vocab_init, NUM_OOV_BUCKETS)

  return table

def build_model(encoder_vocab, decoder_vocab):

  encoder_embedding = tf.keras.layers.Embedding(input_dim=(len(encoder_vocab) + NUM_OOV_BUCKETS) , output_dim=N_EMBEDDING, mask_zero=True)

  decoder_embedding = tf.keras.layers.Embedding(input_dim=(len(decoder_vocab) + NUM_OOV_BUCKETS) , output_dim=N_EMBEDDING, mask_zero=True)

  enc_vocab_size = len(encoder_vocab)

  dec_vocab_size = len(decoder_vocab)

  model_ENCODER = Encoder(encoder_embedding, encoder_vocab)

  encoder_i_tensor = tf.keras.layers.Input (shape= ( None, 1 ) , name = 'encoder_i')

  encoder_o_tensor = model_ENCODER(encoder_i_tensor)

  model_DECODER = Decoder(decoder_embedding, decoder_vocab)

  decoder_o_tensor = model_DECODER(encoder_o_tensor)

  lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
      initial_learning_rate=LEARNING_RATE,
      decay_steps=LEARNING_DECAY_STEPS,
      decay_rate=LEARNING_DECAY_RATE
  )

  if  INFER:
  
    model = tf.keras.Model(encoder_i_tensor, decoder_o_tensor)

    model.load_weights(WEIGHTS_FILENAME)

  else :
  
    model = tf.keras.Model(encoder_i_tensor, decoder_o_tensor)

    model.compile (
      loss = 'sparse_categorical_crossentropy',
      optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
    )

  return model


@tf.keras.utils.register_keras_serializable()
class Encoder(tf.keras.Model): 

  def __init__(self, enc_embed = None, enc_vocab = None, units = NEURONS_LSTM, **kwargs) :

    super().__init__(**kwargs)

    self.units = units

    self.enc_vocab = enc_vocab

    self.enc_embed = enc_embed

    self.encoder_2 = tf.keras.layers.Bidirectional(
      tf.keras.layers.LSTM( self.units , return_sequences=True  , return_state=True , kernel_initializer=tf.keras.initializers.GlorotUniform(seed=2), recurrent_initializer=tf.keras.initializers.GlorotUniform(seed=2) ),
        merge_mode= 'ave',
    )

    self.encoder_gru_2 = tf.keras.layers.Bidirectional(
      tf.keras.layers.GRU( self.units, return_sequences=True, return_state=True, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=2),recurrent_initializer=tf.keras.initializers.GlorotUniform(seed=2))
    )

  def call(self, x):

    if isinstance(self.enc_embed, tf.keras.layers.Embedding):

      print('ENCODER BATCH', x)

      test = self.enc_embed(x)

    else:

      # handle serialized data 
      
      emb = tf.keras.utils.deserialize_keras_object(self.enc_embed)

      test = emb(x)

    test = tf.squeeze(test, 2)

    if USE_GRU:

      e, h, h02, = self.encoder_gru_2(test)

      return e, h, h02, tf.constant([]), tf.constant([])

    else:

      test = self.encoder_2(test)

      e,h0,c0, h02, c02 = test

      return e, h0,c0, h02, c02

  def get_config(self):

    base_config = super().get_config()

    return {
        
      **base_config,

      'units': self.units,

      'enc_vocab' : self.enc_vocab ,

      'enc_embed': tf.keras.layers.serialize(self.enc_embed),

    }

  @classmethod
  def from_config( cls, config):

      return cls(**config)

  def build (self, input_shape):

    return super().build(input_shape)

@tf.keras.utils.register_keras_serializable()
class Decoder( tf.keras.Model):

  def __init__(self, dec_embed, dec_vocab, units = NEURONS_LSTM, **kwargs) :

    super().__init__(**kwargs)

    self.units = units

    self.dec_vocab = dec_vocab # len(self.dec_vocab)

    self.dec_embed = dec_embed

    self.dense_decoder_softmax = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(len(self.dec_vocab), activation='softmax'))

    self.ltsm_cell = tf.keras.layers.LSTMCell(
        self.units * 2,
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=32),
        recurrent_initializer=tf.keras.initializers.GlorotUniform(seed=32)  ,
    )

    self.gru_cell = tf.keras.layers.GRUCell(
        self.units * 2,
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=32),
        recurrent_initializer=tf.keras.initializers.GlorotUniform(seed=32)  ,
    )

    self.gru = tf.keras.layers.GRU(128)

    self.dense1 = tf.keras.layers.Dense(1)

    self.dense_score = tf.keras.layers.TimeDistributed(self.dense1)

    self.dense_decoder_softmax_align_out = (tf.keras.layers.Dense(MAX__TIME_SAMPLES, activation='softmax' ))

    self.softmax = tf.keras.layers.Softmax()

    self.out_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense( len(self.dec_vocab), activation='softmax'))

    self.out_layer_no_td = (tf.keras.layers.Dense(len(self.dec_vocab), activation='softmax'))

    self.ltsm_out = tf.keras.layers.LSTM(len(self.dec_vocab), return_sequences=True, activation='softmax')

    self.concat_layer = tf.keras.layers.Concatenate(axis=1)

    self.weights_all = tf.Variable(tf.zeros((BATCH_SIZE, MAX__TIME_SAMPLES, MAX__TIME_SAMPLES, 1)))

  def ltsm_cell_block(self, x , h, c = None):

    # concatenative attention #

    # x - 8 x 6 x 512

    # h - 8 x 512

    # c - 8 x 512

    if not USE_GRU:

      concat_state = tf.concat( (h,c) , -1)  # 8 x 1024

    else:

      concat_state = tf.concat( (h, c ) , -1)  # 8 x 512

    concat_state = tf.expand_dims(concat_state, 1) # 8 x 1 x 1024

    concat_state_repeat= tf.concat((concat_state, concat_state, concat_state, concat_state, concat_state,concat_state), axis=1) # 8 x 6  x 512

    x_align_concat = tf.concat((x, concat_state_repeat), axis=-1) # 8 x 6 x 1536

    x_align_concat_dense_scores = self.dense1 (x_align_concat)  # 8 x 6 x 1

    x_align_concat_dense_scores = tf.squeeze(x_align_concat_dense_scores, -1) # 8 x 6

    weights = self.dense_decoder_softmax_align_out(x_align_concat_dense_scores) # 8 x 6  softmax METHOD 2

    weights_ext = tf.expand_dims(weights, -1) # 8 x 6 x 1

    weights_mult_enc = tf.multiply(weights_ext, x) # 8 x 6 x 512

    weights_mult_enc_sum = tf.reduce_sum(weights_mult_enc, axis=1) # 8 x 512

    if not USE_GRU:

      y_output, (h_short_term, c_long_term) = self.ltsm_cell(weights_mult_enc_sum, states=[h, c])

      return y_output , (h_short_term, c_long_term), weights

    else:

      y_output, h_short_term = self.gru_cell(weights_mult_enc_sum, states=h)

      return y_output , h_short_term, weights

  def call(self, x):

    if not USE_GRU:

      e, e_h0, e_c0, e_h02, e_c02  = x

      h_encoder = tf.concat( (e_h0, e_h02 ) , -1)

      c_encoder = tf.concat( ( e_c0, e_c02) , -1)

      y0, (h0, c0), w0 = self.ltsm_cell_block(e, h_encoder, c_encoder)

      y1, (h1, c1), w1 = self.ltsm_cell_block(e, h0, c0)

      y2, (h2, c2), w2  = self.ltsm_cell_block(e, h1, c1)

      y3, (h3, c3), w3 = self.ltsm_cell_block(e, h2, c2)

      y4, (h4, c4), w4 = self.ltsm_cell_block(e, h3, c3)

      y5, (h5, c5), w5 = self.ltsm_cell_block(e, h4, c4)

    else :

      e, e_h0, e_h02, e_no, e_no2= x

      h_encoder = tf.concat( (e_h0, e_h02 ) , -1)

      y0, h0, w0  = self.ltsm_cell_block(e, h_encoder)

      y1, h1, w1  = self.ltsm_cell_block(e, h0)

      y2, h2, w2  = self.ltsm_cell_block(e, h1)

      y3, h3, w3  = self.ltsm_cell_block(e, h2)

      y4, h4, w4  = self.ltsm_cell_block(e, h3)

      y5, h5, w5  = self.ltsm_cell_block(e, h4)

    y0 = tf.expand_dims(y0, 1)
    y1 = tf.expand_dims(y1, 1)
    y2 = tf.expand_dims(y2, 1)
    y3 = tf.expand_dims(y3, 1)
    y4 = tf.expand_dims(y4, 1)
    y5 = tf.expand_dims(y5, 1)

    din_distrbuted = self.concat_layer([y0, y1, y2, y3, y4, y5]) # batch x 6 x 512

    din_distrbuted = self.out_layer(din_distrbuted) # batch x 6 x 67

    # capture attention weights 

    w0 = tf.expand_dims(w0, 1)
    w1 = tf.expand_dims(w1, 1)
    w2 = tf.expand_dims(w2, 1)
    w3 = tf.expand_dims(w3, 1)
    w4 = tf.expand_dims(w4, 1)
    w5 = tf.expand_dims(w5, 1)

    w = tf.concat( (w0, w1, w2, w3, w4, w5) , 1)

    w = tf.expand_dims( w, -1)

    length = tf.shape(w)[0]

    self.weights_all[:length].assign(w)

    return din_distrbuted

  def get_config(self):

    base_config = super().get_config()

    return {

      **base_config,

      'units': self.units,

      'dec_vocab':self.dec_vocab,

      'dec_embed': tf.keras.layers.serialize(self.dec_embed),

    }

  @classmethod
  def from_config( cls, config):

      return cls(**config)

  def build (self, input_shape):

    return super().build(input_shape)

# @tf.keras.utils.register_keras_serializable()
class Interference_cb (tf.keras.callbacks.Callback):

  def __init__(self, punctuation_cb, encoder_table, encoder_vocab, decoder_vocab) -> None:

    super().__init__()

    self.punctuation_cb = punctuation_cb

    self.encoder_table = encoder_table

    self.encoder_vocab = encoder_vocab

    self.decoder_vocab = decoder_vocab

  def on_epoch_end(self, epoch = 0, logs=None):

    model = self.model

    writer = tf.summary.create_file_writer(LOG_DIR)
    
    with writer.as_default():

      e = ["November 10 1985", "January 18 2000", "December 23 2010"] # encoder date vectors

      for index, e_selected in enumerate(e):

        e_data = self.punctuation_cb ([e_selected]) # ignore punctuation

        current_encoder_in = e_data[0] #

        test_vector_e = test_vector(current_encoder_in, self.encoder_table, MAX__TIME_SAMPLES, reverse=False) #[ [x],[x],[x]] encoded date

        test_vector_e = tf.cast(test_vector_e, tf.int64)

        test_vector_e = tf.expand_dims(test_vector_e, 0)

        predictions = model(test_vector_e)

        word0 = self.get_word ( tf.argmax(predictions[0][0]) )

        word1 = self.get_word ( tf.argmax(predictions[0][1]) )

        word2 = self.get_word ( tf.argmax(predictions[0][2]) )

        word3 = self.get_word ( tf.argmax(predictions[0][3]) )

        word4 = self.get_word ( tf.argmax(predictions[0][4]) )

        word5 = self.get_word ( tf.argmax(predictions[0][5]) )

        print([word0, word1, word2, word3, word4, word5])
        
        # log translated text and attention weights to tensorboard 

        tf.summary.text('Guess', [ word0, word1, word2, word3, word4, word5 ], step=epoch)
        
        if index == 0:

          tf.summary.image("Attention " + str(index), model.layers[2].weights_all.value(), step=0) # log once !

    %reload_ext tensorboard

  def get_word(self, word_index ):

    if word_index == len(self.decoder_vocab): # OOV hit

      word = self.decoder_vocab[word_index - 1]

    else:

      word = self.decoder_vocab[word_index]

    return word
  
  def get_config(self):
    
    base_config = super().get_config()

    return {
        **base_config,
        'punctuation_cb':self.punctuation_cb,
        'encoder_table':self.encoder_table,
        'encoder_vocab':self.encoder_vocab,
        'decoder_vocab':self.decoder_vocab
    }

with tf.device('/device:GPU:0'):

  # --- Generate Dates

  enc_data = dates(N_DATES)

  decoder = np.vectorize(to_iso_8601)

  dec_data = decoder(enc_data)

  # print(dec_data)

  n_instances = len(enc_data)

  #  -- Encoder

  func_remove_puntuation = np.vectorize(lambda x: re.sub(r'[^\w\s]', '', x )) # all but space and words are replaced with white space

  enc_data = func_remove_puntuation(enc_data)

  func_split = np.vectorize(lambda x: x.split())

  counter_encoder = Counter()

  for x in (enc_data):

    counter_encoder.update((x.split()))

  encoder_vocab = ['<pad>', '<sos>',''] + list(counter_encoder.keys()) + ['<unk>']

  encoder_table = gen_lookup(encoder_vocab)

  encoder_words = tf.constant(encoder_vocab)

  #  -- Decoder

  func_gap_decoder = np.vectorize(lambda x: re.sub(r'(\d+)(-)(\d+)(-)(\d+)', r" <sos> \1 \2 \3 \4 \5", x ))

  dec_data = func_gap_decoder(dec_data)

  print(enc_data)

  print(dec_data)

  counter_decoder = Counter()

  for x in (dec_data):

    counter_decoder.update((x.split()))

  decoder_vocab =  list(counter_decoder.keys())

  decoder_vocab = ['<pad>'] + (((decoder_vocab))) + ['', '<unk>']

  print('Encoder', encoder_vocab)

  print('Decoder', decoder_vocab)

  decoder_table = gen_lookup(decoder_vocab)

  decoder_words = tf.constant(decoder_vocab)

  # --- Encoder Decoder

  encode_ds  = tf.data.Dataset.from_tensor_slices(enc_data)

  if USE_ATTENTION or USE_BIDIRECTIONAL:

    encode_ds_x = encode_ds.map(lambda x: tf.strings.split(x)) # split and reverse encoder input

  else :

    encode_ds_x = encode_ds.map(lambda x: tf.strings.split(x)[::-1]) # split and reverse encoder input

  encode_ds_x = encode_ds_x.map(lambda x: encoder_table.lookup(x)) # lookup

  decode_ds = tf.data.Dataset.from_tensor_slices(dec_data)

  decode_ds = decode_ds.map(lambda x: tf.strings.split(x)) # split and reverse encoder input

  decode_ds = decode_ds.map(lambda x: decoder_table.lookup(x)) # lookup

  decode_ds_x = decode_ds.map(lambda x: x[:-1])

  decode_ds_y = decode_ds.map(lambda x: x[1:])

  decode_ds_targets = decode_ds.map(lambda x: x)

  encode_ds_labels = encode_ds_x.map(lambda x: tf.concat( (x, tf.constant( [0, 0, 0] , dtype=tf.int64 ) )  , 0 ))

  ds = tf.data.Dataset.zip( (encode_ds_x, decode_ds_x ) )

  ds = tf.data.Dataset.zip( (encode_ds_labels,  decode_ds_targets) )

  size_90_percent = int(0.90 * n_instances)

  n_instances_true = n_instances - size_90_percent

  ds_train = ds  #s.take(size_90_percent)

  ds_val = ds.skip(size_90_percent)

  ds_train = ds_train.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

  ds_val = ds_val.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def get_word( word_index , decoder_vocab):

  if word_index >= len(decoder_vocab): # OOV hit

    word = decoder_vocab[word_index - 1]

  else:

    word = decoder_vocab[word_index]

  return word

with tf.device('/device:GPU:0'):

  checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(WEIGHTS_FILENAME, save_weights_only=True, save_best_only=True, mode="max")

  early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=30, restore_best_weights=False)

  ds_train = ds_train.unbatch().map(lambda  record_x, record_y: ( ( record_x[..., None] ),  (record_y[..., None] )  )    )

  ds_val = ds_val.unbatch().map(lambda  record_x, record_y: ( ( record_x[..., None] ),  (record_y[..., None] )  )    )

  if DEBUG:

    ds_train = ds_train.take(100).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    ds_val = ds_val.take(100).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

  else :

    ds_train = ds_train.repeat(FACTOR).shuffle(n_instances * FACTOR).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    ds_val = ds_val.batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

  # for x, y in ds_train.take(1):

  #   print(x)

  #   print(y)

  if not INFER:
    
    model = build_model(encoder_vocab, decoder_vocab)
    
    print(model.summary())

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR)

    gpus = tf.config.experimental.list_physical_devices('GPU')

    cpus = tf.config.experimental.list_physical_devices('CPU')

    print('CPUs', cpus)

    print('GPUS', gpus)

    print( 'EMBED_ DECODER_LSTM', N_EMBEDDING, NEURONS_LSTM)

    history = model.fit(ds_train, epochs = N_EPOCHS,

      validation_data=ds_val,

      callbacks = [

        checkpoint_cb,

        early_stopping_cb,
        
        tensorboard_callback,

        Interference_cb(func_remove_puntuation,encoder_table, encoder_vocab, decoder_vocab)

      ]

    )

  else:

    e = ["September 01, 1987"] # encoder date vectors

    e_data = func_remove_puntuation (e) # ignore punctuation

    current_encoder_in = e_data[0] #

    print(current_encoder_in)

    test_vector_e = test_vector(current_encoder_in, encoder_table, MAX__TIME_SAMPLES, reverse=False) #[ [x],[x],[x]] encoded date

    test_vector_e = tf.cast(test_vector_e, tf.int64)

    test_vector_e = tf.expand_dims(test_vector_e, 0)

    model = build_model( encoder_vocab, decoder_vocab )

    print('INFERRED ', model.summary())

    predictions = model(test_vector_e)

    word0 = get_word ( tf.argmax(predictions[0][0]), decoder_vocab )

    word1 = get_word ( tf.argmax(predictions[0][1]) , decoder_vocab)

    word2 = get_word ( tf.argmax(predictions[0][2]) , decoder_vocab)

    word3 = get_word ( tf.argmax(predictions[0][3]) , decoder_vocab)

    word4 = get_word ( tf.argmax(predictions[0][4]) , decoder_vocab)

    word5 = get_word ( tf.argmax(predictions[0][5]) , decoder_vocab)

    print( 'ESTIMATE' , word0, word1, word2,  word3, word4, word5)
