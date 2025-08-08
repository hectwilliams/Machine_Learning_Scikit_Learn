'''

Filename: 10_implement.py

Creator: Hector Williams 

Repo: https://github.com/hectwilliams/Machine_Learning_Scikit_Learn

Description: 

  Download Movie Review Dataset , splits data, pack data into datasets, and create a model with TextVectorization and Embedding to encode the strings 

  Train model 

  Eval 

  Note: Most important element is my pipeline and enabling AutoGraphing for some function to optimize code 
'''
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt 
import tensorflow as tf
import numpy as np 
import os 
import requests
import regex
import tarfile

FETCH_MOVIE_DATASET = False # enable to download files to CWD and run script  

def fetch_dataset():

  site_url = "https://ai.stanford.edu/~amaas/data/sentiment/"

  response = requests.get(site_url)

  if response.status_code == 200:

    html = response.text

    tars = regex.findall(r'\".+\.gz\"', html) # find tar.gz file 

    ref_file = tars[0][1:-1]

    url_tar_path = os.path.join(site_url, ref_file)

    local_tar_path = os.path.join(os.getcwd(), 'movies_review.tar.gz')

    # validate link 
    if requests.head(url_tar_path).status_code == 200:

      local_file_staging = os.path.join( os.getcwd(), 'movie_reviews' )
      
      try:
        
        os.makedirs(local_file_staging)

      except:

        pass 

      tar_response = requests.get(url_tar_path, stream=True)

      with open(local_tar_path, 'wb') as f:

        f.write(tar_response.raw.read())

      tgz = tarfile.open(local_tar_path)
      
      tgz.extractall(path=local_file_staging)

      tgz.close() 

      os.remove(local_tar_path)


# PIPELINES 

class Stage1(tf.keras.Layer):

  def __init__(self, s_id, s_path, **kwargs):
    
    self.s_id = s_id

    self.path_ = s_path 

    self.prefix = self.path_ + '/' + self.s_id 

    super().__init__(**kwargs) 

  @tf.function
  def call(self, X=None):
    pos_names = tf.io.gfile.listdir ( self.prefix + '/' + 'pos'  )

    pos_names_full =  tf.map_fn(lambda name: tf.strings.join([ self.prefix, 'pos', name],separator='/' ) , elems= tf.constant(pos_names))


    neg_names = tf.io.gfile.listdir ( self.prefix + '/' + 'neg'  )

    neg_names_full =  tf.map_fn(lambda name: tf.strings.join([ self.prefix, 'neg', name],separator='/' ) , elems= tf.constant(neg_names))

    return pos_names_full, neg_names_full

class Stage2(tf.keras.Layer):

  @tf.function
  def call(self, X=None):

    pos, neg = X 

    pos = tf.map_fn(lambda filepath: tf.io.read_file(filepath) + b'1' , elems=pos) # last byte stores the label

    neg = tf.map_fn(lambda filepath: tf.io.read_file(filepath) + b'0' , elems=neg) 

    return pos, neg

class Stage3(tf.keras.Layer):

  @tf.function
  def call(self, X=None):

    pos, neg = X 

    data = tf.concat( [pos, neg] , axis=0 )

    return data

@tf.function
def tuplize(bytestring):
  
  statement = tf.strings.substr(bytestring, 0, tf.strings.length(bytestring) - 1)
  
  label_byte = tf.strings.substr(bytestring, tf.strings.length(bytestring) - 1, 1)
  
  label =  -tf.strings.unicode_decode(label_byte, "UTF-8") + 49
  
  return statement, label

class Stage4(tf.keras.Layer):

  @tf.function
  def call(self, X):

    return tf.random.shuffle(X)

class Stage5(tf.keras.Layer):
    
  @tf.function
  def call(self, X ):

    ds = tf.data.Dataset.from_tensor_slices(X) # single element tensor housings list variables 

    ds = ds.map(tuplize, num_parallel_calls=6).shuffle(len(ds)) # dataset datatype = (string, int)

    return ds

def file_pipeline(set_type ):
  '''
    input preprocessing pipeline 

    Arguments:

      set_type - dataset name. Two possible strings [train, test] ( Default = train )

      text_vect - required for set_type = train 

    Returns:

      Dataset 
  '''

  if set_type in ['train', 'test']:

    path_ = os.path.join(os.getcwd(), 'movie_reviews', 'aclImdb') 

    z = Stage1(set_type, path_)()

    z = Stage2()(z)

    z = Stage3()(z)

    z = Stage4()(z)

    ds = Stage5()(z)

    return ds 

  else:

    raise Exception('two possible choices for argument set_type - [train, test]')


# BUILD MODEL ( SUBCLASS VERSION )

class TextVectorization2(tf.keras.layers.TextVectorization):
  
  def __init__(self, ds, **kwargs):

    self.dataset = ds 

    super().__init__(**kwargs) 

    self.adapt(self.dataset)

class Embedding2(tf.keras.layers.Embedding):

  def __init__(self, input_dim, output_dim,  **kwargs):

    self.out_dim = output_dim

    self.input_dim = input_dim

    super().__init__(input_dim, output_dim, **kwargs)


class ReviewsModel(tf.keras.Model):

  def __init__(self, textvectorization, embedding, **kwargs ):

    super().__init__(**kwargs)

    self.layer_textvectorization = textvectorization

    self.layer_embedding = embedding 

    self.layer_hidden_Dense_1 = tf.keras.layers.Dense(50,activation= tf.keras.activations.relu  ) 
    
    self.layer_hidden_Dense_2 = tf.keras.layers.Dense(50,activation= tf.keras.activations.relu  ) 

    self.layer_hidden_Dense_3 = tf.keras.layers.Dense(50,activation= tf.keras.activations.relu  ) 

    self.out = tf.keras.layers.Dense(1, activation = tf.keras.activations.sigmoid  )

  def call(self, inputs): 
                          
    z = self.layer_textvectorization(inputs) # input -> batch_size  x 1
    
    self.textvectorizer_aux = z # batch_size x 3

    z = self.layer_embedding(z)

    z = self.layer_hidden_Dense_1(z)

    z = self.layer_hidden_Dense_2(z)

    z = self.layer_hidden_Dense_3(z)

    z_out = self.out(z) # batch_size x 1

    return z_out
  
  def compute_output_shape(self, batch_input_shape):

    return batch_input_shape
  
  def get_config(self):

    base_config = super.get_config()

    return {**base_config, 'textvectorization': tf.keras.layers.serialize(self.layer_textvectorization), 'embedding': tf.keras.layers.serialize(self.layer_embedding) }

def run():

  '''
    custom training loop
  '''
  
  ds_train = file_pipeline('train')

  ds_test = file_pipeline('test')

  ds_val = ds_test.take(15000)

  ds_test = ds_test.skip(15000)

  ds_comments = ds_train.map(lambda cmt, _ : cmt )

  textvectorizer = TextVectorization2(ds_comments)
    
  embedding = Embedding2(input_dim = textvectorizer.vocabulary_size() + 2  , output_dim = 8)  
  
  epochs = tf.constant(5)

  batch_size = tf.constant(32, dtype=tf.int64)

  optimizer = tf.keras.optimizers.SGD( learning_rate = 0.010 )

  mean_loss = tf.keras.metrics.Mean()

  best_loss = tf.Variable(0, dtype=tf.float32)

  model = build_model(textvectorizer, embedding)
  
  writer = tf.summary.create_file_writer(os.path.join(os.getcwd(), 'tensorboard_logs'))

  with writer.as_default():

    for epoch in tf.range(1,epochs +1):

      for step, (x_batch, y_batch) in tf.data.Dataset.enumerate(ds_train.shuffle(buffer_size=ds_train.cardinality()).batch(batch_size)):
        
        with tf.GradientTape() as tape:
        
          y_pred_batch, text_vect_aux_batch = model(x_batch, training=True)

          filtered_mean_aux = tf.boolean_mask(text_vect_aux_batch, text_vect_aux_batch <= 100) # filters histogram
          
          # zeros_count = tf.reduce_sum(tf.cast(tf.equal(filtered_mean_aux, 0), tf.int32))

          # ones_count = tf.math.count_nonzero(filtered_mean_aux)

          tf.summary.histogram('Text-Vectorize0', filtered_mean_aux, step = step)

          loss = tf.reduce_mean(tf.keras.losses.BinaryCrossentropy() (y_batch, y_pred_batch))

          gradients = tape.gradient(loss, model.trainable_variables)
          
          optimizer.apply_gradients(zip( gradients, model.trainable_variables))

        mean_loss(loss) 
      
        print(f'\r EPOCH -- {epoch}  STEP -- {step}  LOSS -- {loss}', end='')

      print('AVG LOSS -- {}'.format(mean_loss.result()))

      print('EVALUATE VALIDATION SET   {} ', model.evaluate(ds_val))

      if best_loss < mean_loss.result():

        model.save('reviews.keras')

        best_loss.assign = mean_loss.result()


if __name__ == '__main__':

  if FETCH_MOVIE_DATASET:

    fetch_dataset()
  
  run()