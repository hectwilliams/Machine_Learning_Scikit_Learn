'''

Filename: 10_implement.py

Creator: Hector Williams 

Repo: https://github.com/hectwilliams/Machine_Learning_Scikit_Learn

Description: 

  Download Movie Review Dataset , splits data, pack data into datasets, and create a model with TextVectorization and Embedding to encode the strings 

  Train model 

  Eval 

  Note: custom pipelines created for readable and efficient code ( efficient using tf.function  to optimize pipeline stages)
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

    self.adapt(ds)

class Embedding2(tf.keras.layers.Embedding):

  def __init__(self, input_dim, output_dim,  **kwargs):

    self.output_dim = output_dim

    self.input_dim = input_dim

    super().__init__(input_dim, output_dim, **kwargs)



class ReviewsModel(tf.keras.Model):

  def __init__(self, textvectorization, embedding, **kwargs ):

    super().__init__(**kwargs)

    self.layer_textvectorization = textvectorization

    self.layer_embedding = embedding 
    
    self.layer_avg_embedding = tf.keras.layers.Lambda(lambda matrix: tf.sqrt( tf.cast(  tf.shape(matrix)[1] , tf.float32)) * tf.reduce_mean(matrix, axis = 1) , output_shape=(embedding.output_dim,)  ) # product of avg-embedding and number of words in comment 

    self.layer_hidden_Dense_1 = tf.keras.layers.Dense(10   , activation= tf.keras.layers.ELU(), kernel_initializer=tf.keras.initializers.HeUniform(seed=32) ) 
    
    self.layer_hidden_Dense_2 = tf.keras.layers.Dense(400   , activation= tf.keras.layers.ELU(), kernel_initializer=tf.keras.initializers.HeUniform(seed=32) ) 

    self.layer_hidden_Dense_3 = tf.keras.layers.Dense(400   , activation= tf.keras.layers.ELU(), kernel_initializer=tf.keras.initializers.HeUniform(seed=32) ) 
    
    self.layer_hidden_Dense_4 = tf.keras.layers.Dense(10   , activation= tf.keras.layers.ELU(), kernel_initializer=tf.keras.initializers.HeUniform(seed=32) ) 

    self.zout = tf.keras.layers.Dense( 1, activation='sigmoid'  , kernel_initializer=tf.keras.initializers.GlorotNormal(seed=32))
    

  def call(self, inputs): 
                          
    z = self.layer_textvectorization(inputs) 
    
    # self.textvectorizer_aux = z # side channel for tensorboard 

    z = self.layer_embedding(z)
    
    z = self.layer_avg_embedding(z) 
    
    z = self.layer_hidden_Dense_1(z)
    
    z = self.layer_hidden_Dense_2(z)

    z = self.layer_hidden_Dense_3(z)
    
    z = self.layer_hidden_Dense_4(z)

    return self.zout(z)   

  def build (self, batch_input_shape):

    super().build(batch_input_shape)

  def compute_output_shape(self, batch_input_shape):

    return batch_input_shape
  
  def get_config(self):

    base_config = super().get_config()

    return {**base_config, 'layer_textvectorization': tf.keras.layers.serialize(self.layer_textvectorization), 'layer_embedding': tf.keras.layers.serialize(self.layer_embedding) }


class LearningRateScheduler2(tf.keras.callbacks.LearningRateScheduler):
  
  def __init__(self, lr, s = 40, **kwargs):

    def lr_scheduler_dec(func):
      
      def lr_scheduler_wrapper(lr_, s_):

        def keras_lr_scheduler (epoch):

          return lr_ * 0.1 **(epoch/s_)

        return keras_lr_scheduler
        
      return lr_scheduler_wrapper
      
    @lr_scheduler_dec
    def lr_scheduler(lr, s): tf.no_op() 

    super().__init__(lr_scheduler(lr, s), **kwargs)


def run():
  
  epochs = 10

  size_dataset = -1 

  size_batch = 8

  ds_train = file_pipeline('train').take(size_dataset)

  ds_test = file_pipeline('test')

  ds_val = ds_test.take(15000)

  ds_test = ds_test.skip(15000)  

  ds_comments = ds_train.map(lambda cmt, _ : cmt )

  textvectorizer = TextVectorization2(ds_comments)

  embedding = Embedding2(textvectorizer.vocabulary_size() + 2  , 128)  

  model = ReviewsModel(textvectorizer, embedding)

  model.compile ( loss = tf.keras.losses.BinaryCrossentropy(), optimizer = tf.keras.optimizers.SGD(learning_rate=0.03, clipvalue=200, decay=1/4), metrics=[tf.keras.metrics.BinaryAccuracy() ] )

  checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('chapter13_movies.keras', save_best_only=True)

  tensorboard_cb = tf.keras.callbacks.TensorBoard(os.path.join(os.getcwd(), "tensorboard_logs"))

  learning_rate_cb = LearningRateScheduler2(0.04) 
  
  ds_train_batch = ds_train.batch(size_batch) 

  ds_val_batch = ds_val.batch(size_batch)

  ds_test_batch = ds_test.batch(size_batch)

  model.fit(ds_train_batch, validation_data = ds_val_batch, epochs=epochs , callbacks=[checkpoint_cb, tensorboard_cb, learning_rate_cb])
    
  model.evaluate(ds_test_batch)

if __name__ == '__main__':

  if FETCH_MOVIE_DATASET:

    fetch_dataset()
  
  run()