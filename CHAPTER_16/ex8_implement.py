
''' 

  Embedded Reber Grammar Model, creates graph, learns which possible paths are valid, and infer on string sequence 

  Test vectors estimate results after 14 epochs 
   
    [
      'BT',  # invalid embedded reber           ( 0.238 %  probability  )
      'BBDACBT', # invalid embedded reber       ( 0.15 % probability  )
      'BTSSXXTVVE', # invalid embedded reber    ( 0.074 % probability  )
      'BTBTSSXXVVETE',  # valid embedded reber  ( 86 % probability     )
      'BTSSPXSE', # invalid embedded reber      ( 0.0836 % probability  )
      'BTBPTVVETE',  # valid embedded reber     ( 56.25 % probability   )
    
    ]    

'''

import os
import re
import regex
import numpy as np
import tensorflow as tf
from collections import Counter

np.random.seed(1)
N_STEPS = 256
N_FEATURES = 1 #( 1 character per step)
TRAIN = 1
EMBEDDING = 128
UNITS = 64
NUM_OOV_BUCKETS = 10
N_EPOCHS = 30

class RebelModel(tf.keras.Model):

  def __init__(self, vocab_length, num_oov_buckets, embedding, units = UNITS , **kwargs):

    super().__init__(**kwargs)

    self.vocab_length = vocab_length

    self.num_oov_buckets = num_oov_buckets

    self.embedding = embedding

    self.units = units

    self.lstm_layer1 = tf.keras.layers.LSTM(units=self.units, return_sequences=True)

    self.lstm_layer2 = tf.keras.layers.LSTM(units=self.units, return_sequences=True)

    self.lstm_layer3 = tf.keras.layers.LSTM(units=self.units  )

    self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

  def call(self, inputs):

    z = tf.squeeze(self.embedding(inputs), 2)

    z = self.lstm_layer1(z)
    
    z = self.lstm_layer2(z)

    z = self.lstm_layer3(z)

    z = self.output_layer(z)

    return z

  def get_config(self):

    base_config = super().get_config()

    return {**base_config, 'vocab_length': self.vocab_length , 'num_oov_buckets': self.num_oov_buckets ,  'embedding': self.embedding}

  def compute_output_shape(self, batch_input_shape):

    return (batch_input_shape[0], 1)

# --- Build Graph 

class ReberNode:

  def __init__(self, id = 1):

    self.table = {}

    self.ingress = [] # ingress states

    self.egress = []  # egress states

    self.id = id

  def add_state(self, state, node):

    self.table[state] = node

  def add_ingress(self, state):

    self.ingress.append(state)

  def add_egress(self, state):

    self.egress.append(state)

class Embedded_Reber_Graph:

  def __init__(self):

    self.nodes = [ReberNode(i) for i in range(0, 20)]

    self.root = self.nodes[0]

  def __call__(self, *args, **kwargs):

    ''' builds graph '''

    for node in self.nodes:

      if node.id == 0:

        node.add_state('B', self.nodes[1])

        node.add_egress('B')

      if node.id == 1:

        node.add_state('T', self.nodes[2])

        node.add_state('P', self.nodes[3])

        node.add_ingress('B')

        node.add_egress('T')

        node.add_egress('P')

      if node.id == 2:

        node.add_state('B', self.nodes[4])

        node.add_egress('B')

        node.add_ingress('T')

      elif node.id == 3:

        node.add_state('B', self.nodes[10])

        node.add_ingress('P')

        node.add_egress('B')

      elif node.id == 4:

        node.add_state('T', self.nodes[5])

        node.add_state('P', self.nodes[6])

        node.add_ingress('B')

        node.add_egress('T')

        node.add_egress('P')

      elif node.id == 5:

        node.add_state('S', self.nodes[5])

        node.add_state('X', self.nodes[7])

        node.add_ingress('T')

        node.add_ingress('S')

        node.add_egress('X')

      elif node.id == 6:

        node.add_state('T', self.nodes[6])

        node.add_state('V', self.nodes[8])

        node.add_ingress('T')

        node.add_ingress('P')

        node.add_ingress('X')

        node.add_egress('V')

      elif node.id == 7:

        node.add_state('X', self.nodes[6])

        node.add_state('S', self.nodes[9])

        node.add_ingress('X')

        node.add_ingress('P')

        node.add_egress('X')

        node.add_egress('S')

      elif node.id == 8:

        node.add_state('P', self.nodes[7])

        node.add_state('V', self.nodes[9])

        node.add_ingress('V')

        node.add_egress('P')

        node.add_egress('V')

      elif node.id == 9:

        node.add_state('E', self.nodes[16])

        node.add_ingress('V')

        node.add_ingress('S')

        node.add_egress('E')


      elif node.id == 10:

        node.add_state('T', self.nodes[11])

        node.add_state('P', self.nodes[12])

        node.add_ingress('B')

        node.add_egress('T')

        node.add_egress('P')

      elif node.id == 11:

        node.add_state('S', self.nodes[11])

        node.add_state('X', self.nodes[13])

        node.add_ingress('S')

        node.add_ingress('T')

        node.add_egress('X')


      elif node.id == 12:

        node.add_state('T', self.nodes[12])

        node.add_state('V', self.nodes[14])

        node.add_ingress('T')

        node.add_ingress('P')

        node.add_ingress('X')

        node.add_egress('V')

      elif node.id == 13:

        node.add_state('X', self.nodes[12])

        node.add_state('S', self.nodes[15])

        node.add_ingress('X')

        node.add_ingress('P')

        node.add_egress('X')

        node.add_egress('S')

      elif node.id == 14:

        node.add_state('P', self.nodes[13])

        node.add_state('V', self.nodes[15])

        node.add_ingress('V')

        node.add_egress('P')

        node.add_egress('V')

      elif node.id == 15:

        node.add_state('E', self.nodes[17])

        node.add_ingress('S')

        node.add_ingress('V')

        node.add_egress('E')

      elif node.id == 16:

        node.add_state('T', self.nodes[18])

        node.add_ingress('E')

        node.add_egress('T')

      elif node.id == 17:

        node.add_state('P', self.nodes[18])

        node.add_ingress('E')

        node.add_egress('P')

      elif node.id == 18:

        node.add_state('E', self.nodes[19])

        node.add_ingress('T')

        node.add_ingress('P')

        node.add_egress('E')


  def path(self, make_invalid=False):
    ''' traverses random path in graph '''

    result = ''

    node = self.root

    while node != None :

      possible_states = node.table.keys()

      choices = list(possible_states)

      current_node_id = node.id

      all_ids = [node_inner.id for node_inner in self.nodes]

      possible_ids = [ node.table[s].id for s in choices ]

      self.nodes[current_node_id].table.keys()

      if len(choices) != 0:

        # truncate path ( invalidates path )

        if np.random.random() > 0.95 and make_invalid:

          break

        elif np.random.random() > 0.7 and make_invalid:

          # jump to random  node

          jump_id = np.random.randint(low=1, high=len(self.nodes))

          jump_node = self.nodes[jump_id]

          result += str(jump_id) + ('' if not jump_node.ingress  else np.random.choice(jump_node.ingress)  )

          node = jump_node

          if node.id == 19:

            break

        else:

          result += str(node.id) + np.random.choice(choices)

          node = node.table[result[-1]]

      else:

        result += str(node.id)

        node = None

    return result

def is_valid_path(graph: Embedded_Reber_Graph, path:str):

  if len(path.strip()) == 0:

    return False

  while path:

    path = path.strip()

    if path.strip() == '19':

      result += '19'

      break

    m = regex.match(r'(\d*)([BTPXSVE]*)(\d+)*', path)

    current_node_id = m.groups()[0].strip()

    current_node = graph.nodes[int(current_node_id)]

    possible_next_states = graph.nodes[int(current_node_id)].table.keys()

    possible_next_ids = [ current_node.table[next_state].id for next_state in possible_next_states ]

    next_state_pred = m.groups()[1].strip() # lookahead

    next_node_id_pred = m.groups()[2].strip() if m.groups()[2] else '-1'

    if not next_node_id_pred:

      return False

    truncate_head_size = m.span()[1] - len(next_node_id_pred)

    if not ((int(next_node_id_pred) in possible_next_ids) and (next_state_pred in possible_next_states)) :

      return False

    path = path[truncate_head_size : ] if next_state_pred else None

  return True

def is_substring(strings,s ):

  for string in strings:

    if string in s:

      return True

  return False

def loop_max_hit(path, n_loop = 5):
  
  "limit number of times a node can be revisited by a node if a loop exist"
  
  if path:

    p = re.sub(r'[^A-Z]', '', path) # regex - match a non captial alpha

    if Counter(p).get(path[-1]) > n_loop:
      
      return True 

  return False 

def all_paths(graph: Embedded_Reber_Graph):

  q = [dict(node=graph.root, path='')]

  mem = {}
  
  node_out_table = {}

  while q:

    record = q.pop(0)

    node = record['node']

    path = record['path']
    
    regex_ret = re.match(r'([0-9]+[A-Z])*(\d+)([A-Z])',path)

    if regex_ret:

      prev_node = regex_ret.groups()[-2]

      prev_state = regex_ret.groups()[-1]

    if len(mem) == 10000:
      
      return np.array(list(mem.keys()))
      
    if node.id == 19:
      
      mem[path ] = True

    else:

      for state in node.table.keys():
        
        if node.id not in node_out_table :
          
          node_out_table[node.id] = {}

        if state not in  node_out_table[node.id]:
        
          node_out_table[node.id][state] = 0 

        node_out_table[node.id][state] += 1

        if not loop_max_hit(path, n_loop=8):

          q.append(dict(node=node.table[state], path = path + str(node.id) + state))

  np.save('paths.npy', list(mem.keys()))


# --- Callback 

class PrintInferenceCB(tf.keras.callbacks.Callback):
  
  "predictions made on test vector "
  
  def on_epoch_end(self, epoch, logs=None):

    input_data = [
      'BT',
      'BBDACBT',
      'BTSSXXTVVE',
      'BTBTSSXXVVETE',
      'BTSSPXSE',
      'BTBPTVVETE',
    ]

    func = np.vectorize(lambda x: ' '.join( list(x) ) )

    v = tf.constant(func(input_data))

    v = tf.strings.split(v)  # split each element(i.e. )

    v = table.lookup(v)

    v = tf.map_fn(lambda x: x[ ..., None] , v)

    v = v.to_tensor()

    predictions =  r_model(v)

    print((predictions))

n_paths = 1000

graph = Embedded_Reber_Graph()

graph()

# --  Generate valid paths 

if not os.path.exists('paths.npy'):
  all_paths(graph)

paths = np.load('paths.npy', allow_pickle=True)

# --  Generate invalid paths 

if not os.path.exists('invalid_paths.npy'):

  invalid_paths = [graph.path(make_invalid=True) for i in range(n_paths)]

  invalid_paths = set(invalid_paths)

  invalid_paths = list(invalid_paths)

  invalid_paths = np.array(invalid_paths)

  np.save('invalid_paths.npy', invalid_paths)

valid_paths = np.load('paths.npy') # TODO rerun generation script

invalid_paths = np.load('invalid_paths.npy') # TODO rerun generation script

#  -- Match Size
min_size = np.minimum(len(valid_paths), len(invalid_paths)) 

invalid_paths = invalid_paths[:min_size]

valid_paths = valid_paths[:min_size]

#  -- Remove node identifiers

a = regex.sub(r'(\d+)', '', valid_paths[0])

remove_digit_func = np.vectorize(lambda x: regex.sub(r'(\d+)', '', x))

split_func = np.vectorize(lambda x: " ".join(x))

valid_paths = (remove_digit_func(valid_paths))

invalid_paths = (remove_digit_func(invalid_paths))

#  -- Build Vocab

counter = Counter()

for path in valid_paths:

  counter.update(list(path))

for path in invalid_paths:

  counter.update(list(path))

words = [  '<pad>', '<sos>','<unk>', ''] +  list(counter.keys())

words_ = words

words = tf.constant(words)

words_ids = tf.range( tf.shape(words)[-1], dtype=tf.int64)

vocab_init = tf.lookup.KeyValueTensorInitializer(words, words_ids)

num_oov_buckets = NUM_OOV_BUCKETS

table = tf.lookup.StaticVocabularyTable(vocab_init, num_oov_buckets)

# -- Preprocess Training Data

func = np.vectorize(lambda x: ' '.join( list(x) ) )

valid_paths = func(valid_paths)

invalid_paths = func(invalid_paths)

ds_good_paths = tf.data.Dataset.from_tensor_slices(valid_paths)

ds_good_paths = ds_good_paths.map(lambda x: (tf.strings.split(x)))

ds_good_paths = ds_good_paths.map(lambda x: (table.lookup(x)))

ds_good_paths = ds_good_paths.map(lambda x: (x, 1))

ds_bad_paths = tf.data.Dataset.from_tensor_slices(invalid_paths)

ds_bad_paths = ds_bad_paths.map(lambda x: (tf.strings.split(x)))

ds_bad_paths = ds_bad_paths.map(lambda x: (table.lookup(x)))

ds_bad_paths = ds_bad_paths.map(lambda x: (x, 0))

n_reber_paths = len(valid_paths) + len(invalid_paths)

train_size = int(0.8 * n_reber_paths)

val_size = int(0.2 * n_reber_paths)

ds_prepad = ds_good_paths.concatenate(ds_bad_paths).shuffle(n_reber_paths, seed=32).repeat(50)

train_ds = ds_prepad.take(train_size)

val_ds = ds_prepad.skip(train_size)

vocab = words

# -- Custom Model (Fit)

train_ds = train_ds.map(lambda x , y : (x[..., None], y))

val_ds = val_ds.map(lambda x , y : (x[..., None], y))

train_ds = train_ds.padded_batch(4)

val_ds = val_ds.padded_batch(4)

keras_filename = 'reber_model3.weights.h5'

input_tensor = tf.keras.layers.Input (shape= (None,1) , dtype=tf.int32, name ='tokenized_data')

embedding = tf.keras.layers.Embedding(input_dim=(len(vocab) + num_oov_buckets) , output_dim=EMBEDDING, mask_zero=True)

r_model = RebelModel(len(vocab) + num_oov_buckets, num_oov_buckets, embedding) # constructor

output_tensor = r_model(input_tensor)

tf.keras.Model(inputs=[input_tensor], outputs=[output_tensor])

r_model.compile (loss='binary_crossentropy', optimizer='adam', metrics=[tf.keras.losses.BinaryCrossentropy()])

if TRAIN or not os.path.exists(keras_filename):

  if os.path.exists(keras_filename):

    r_model.load_weights(keras_filename)

  checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(keras_filename, save_best_only=True, save_weights_only=True)

  early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

  r_model.fit(
    train_ds,
    epochs=N_EPOCHS,
    validation_data = val_ds,
    callbacks= [checkpoint_cb, early_stopping_cb, PrintInferenceCB()])

