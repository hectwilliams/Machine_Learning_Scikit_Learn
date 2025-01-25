#!/usr/bin/env python3

'''



'''

from tensorflow import keras 
import tensorflow as tf  
from functools import partial
import os 
import numpy as np
import csv
import sys 
import time 
from tensorflow.train import BytesList, FloatList, Int64List, Feature, Features, Example
import requests
import tarfile
import re

IMDB_DATASET_URL = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
WORKING_DIR = os.path.join(os.getcwd(), 'movie_review_dataset')

def download_dataset(url, name):
    ''' 
        Download tar gz file 
    '''
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(os.path.join(WORKING_DIR, name) , 'wb') as f:
            f.write(response.raw.read())

def extract_gzip(file_path, extract_path):
    try:
        with tarfile.open(file_path, 'r:gz') as tar:
            tar.extractall(extract_path)
    except tarfile.TarError as e:
        print('tar.gz file extraction error')

try :
    os.makedirs(WORKING_DIR)
except OSError as e:
    print(f'OSERROR:{e.errno} DIRECTORY EXISTS')

zipfile = re.search(r'.+\/(aclImdb_v1\.tar\.gz)', IMDB_DATASET_URL)
tar_filename = zipfile.groups(0)[0]

if not os.path.exists(os.path.join( WORKING_DIR, tar_filename)):
    download_dataset(IMDB_DATASET_URL, tar_filename)

if not os.path.exists(os.path.join(WORKING_DIR, 'aclImdb')):
    extract_gzip( os.path.join( WORKING_DIR, tar_filename), WORKING_DIR )
