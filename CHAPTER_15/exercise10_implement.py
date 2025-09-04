'''
  Trains a model using recurrent Neural Networks to predict the next time step given any varying sequence of time steps. 

  Time steps house a musical 4 note data vector. Future predictions will be a single vector. The is an example of sequence to vector which can be used to create a sequence to sequence generator.  

  Note: Model not trained! Architecture and preprocessing processes can be used to predict next time sample of varying systems.
'''

import tarfile
import os
import asyncio
import tensorflow as tf
import requests

# constants 
MAX_CHORAL_LENGTH = 640 

# functions 
def extract_tgz(tgz_file_path, destination_directory):
    """
    Extracts the contents of a .tgz file to a specified destination directory.

    Args:
        tgz_file_path (str): The path to the .tgz file to be extracted.
        destination_directory (str): The path to the directory where
                                     the contents will be extracted.
    """
    try:
        # Create the destination directory if it doesn't exist
        os.makedirs(destination_directory, exist_ok=True)

        # Open the .tgz file in read-gzip mode
        with tarfile.open(tgz_file_path, "r:gz") as tar:
            # Extract all contents to the destination directory
            tar.extractall(path=destination_directory)
        print(f"Successfully extracted '{tgz_file_path}' to '{destination_directory}'")
    except tarfile.ReadError as e:
        print(f"Error reading tarfile: {e}")
    except FileNotFoundError:
        print(f"Error: The file '{tgz_file_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def fetch_tgz(url):
  target_path = os.path.join(os.getcwd(), 'jsb_chorales.tgz')  # Desired local filename

  try:
      response = requests.get(url, stream=True)
      response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

      with open(target_path, 'wb') as f:
          for chunk in response.iter_content(chunk_size=8192):
              f.write(chunk)
      print(f"Downloaded '{target_path}' successfully.")

  except requests.exceptions.RequestException as e:
      print(f"Error downloading file: {e}")

def xy_chords(csv_filename):
  '''
    Per csv file, generate X - data and Y - Label-Data. Label data is the next row in the csv file.
  '''
  ds_list_single_element = tf.data.Dataset.from_tensor_slices([csv_filename])
  
  dataset_curr = ds_list_single_element.interleave( lambda filename: tf.data.TextLineDataset(filename).skip(1).map( lambda x: (x) ), cycle_length = 1, num_parallel_calls = tf.data.experimental.AUTOTUNE)

  dataset_next = ds_list_single_element.interleave( lambda filename: tf.data.TextLineDataset(filename).skip(2).map( lambda x: (x) ), cycle_length = 1, num_parallel_calls = tf.data.experimental.AUTOTUNE)

  dataset_dummy_pad = tf.data.Dataset.zip (tf.data.Dataset.from_tensor_slices([ tf.constant(b'0,0,0,0')]  )  ) 
  
  dataset_next = dataset_next.concatenate( dataset_dummy_pad )

  # pad to 640 
  
  n_lines = dataset_curr.map(lambda x: 1).reduce(initial_state=0, reduce_func = lambda x,y: x+y)

  n_pad = tf.constant(640, dtype=tf.int32) - n_lines
  
  if n_pad > 0:
    
    padd_ones = tf.constant(b'0,0,0,0')
    
    padd_ones = tf.repeat(padd_ones[None, ...], n_pad,0)

    padd_ones = tf.data.Dataset.from_tensor_slices(padd_ones )

    dataset_curr = padd_ones.concatenate(dataset_curr)
    
    dataset_next = padd_ones.concatenate(dataset_next)

  dataset = tf.data.Dataset.zip((dataset_curr, dataset_next))

  dataset = dataset.map(csvline_to_int32)

  return dataset

def csvline_to_int32(curr, next):

  # fields =  tf.io.decode_csv(line, record_defaults=def_types)
  
  curr_int32 = tf.strings.to_number( tf.strings.split(curr, ',') , out_type=tf.int32)
  
  next_int32 = tf.strings.to_number( tf.strings.split(next, ',') , out_type=tf.int32)

  return curr_int32, next_int32
  
def csv_reader_dataset(dir_name = 'train', n_readers = 1, n_read_threads = 3):
    """

    """
    dir = os.path.join(os.getcwd(), 'jsb_chorales', dir_name)

    list_of_files = os.listdir(dir)

    list_of_files = [os.path.join(dir, file) for file in list_of_files]
    
    datasetList = tf.data.Dataset.from_tensor_slices(list_of_files) # order perserved 
    
    dataset_window = datasetList.window(1)

    ds = dataset_window.map(lambda w: w.map(xy_chords) , num_parallel_calls=3)

    ds = ds.flat_map(lambda x: x)
    
    ds = ds.flat_map(lambda x: x)

    ds = ds.map(lambda x, y: ( tf.reshape(x, (1,4)), tf.reshape(y, (1,4))) )
    
    ds = ds.batch(MAX_CHORAL_LENGTH) # batch 640 1x4 vectors --> 640 x 1 x 4

    ds = ds.map(lambda x, y: (tf.reshape(x, (MAX_CHORAL_LENGTH,4)), tf.reshape(y, (MAX_CHORAL_LENGTH,4))) ) # reshape to 640 x 4

    ds = ds.batch(8).prefetch(1) # batch each time series block 

    return ds 


def get_model():

  """

    Architecture to predict time series. Because we want to predict every step of a series, all RNN cells return sequences.

  """

  z = ii = tf.keras.layers.Input(shape= (None,4) , name ='input')

  z = tf.keras.layers.LSTM(units=30, return_sequences=True) (z)

  z = tf.keras.layers.LSTM(units=30, return_sequences=True) (z)

  z = tf.keras.layers.LSTM(units=30, return_sequences=True) (z)

  z = oo = tf.keras.layers.Dense(4) (z)

  model = tf.keras.Model(inputs=[ii], outputs=[oo])

  return model

async def view_chorales():
  
  await asyncio.sleep(0.5)

  return csv_reader_dataset('train'), csv_reader_dataset('test'), csv_reader_dataset('valid')


async def get_raw_files():
  
  if f'jsb_chorales' not in os.listdir(os.getcwd()):
      
    if f'jsb_chorales.tgz' not in os.listdir(os.getcwd()):
      
      fetch_tgz('https://homl.info/bach')
      
    extract_tgz(os.path.join(os.getcwd(), 'jsb_chorales.tgz'), os.getcwd())
  
  return True


if __name__ == "__main__":
    
    # tasks ( tasks are SYNCHRONOUS )
    
    ds = None 

    status = await asyncio.create_task(get_raw_files())

    if status:
    
      ds = await asyncio.create_task(view_chorales())

    if ds:

      ds_train, ds_test, ds_valid = ds

      model = get_model() 

      model.compile(loss='mse', optimizer = 'adam' )

      model.fit(ds_train, epochs=2, validation_data=ds_valid)

      # predicts the next choral after a single rcvd choral

      notes_1 = tf.constant([ [55,55,55,55] ] , shape = (1,1,4))

      predict_2 = print(model(notes_1))
      
      # predicts the next choral after a two chorals are rcvd 

      notes_2 = tf.constant([ [55,55,55,55] , [65,52,53,55] ] , shape = (1,2,4))
      
      predict_3 = print(model(notes_2))



      

