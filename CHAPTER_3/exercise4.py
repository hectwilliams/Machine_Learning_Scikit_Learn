import os 
import re 
import tarfile
import requests
import numpy as np 
import pandas as pd 
from nltk.stem import PorterStemmer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from classifier_table import table_of_classifiers
from sklearn.metrics import precision_score, recall_score, f1_score

QUICK_N = 100
LONGEST_WORD_SIZE = 45 

class FetchDataset(BaseEstimator, TransformerMixin):

  def __init__(self, mode = 0):

    '''
      
      Fetch data Span Assassin public dataset : https://spamassassin.apache.org/old/publiccorpus/

    '''

    print('Fetching ...')

    file_prefix = [ '20021010', '20030228' ]
    
    url = "https://spamassassin.apache.org/old/publiccorpus/"

    try:

      # download compressed files from spamassassin
      
      html_text = requests.get(url).text
      
      basenames = list(filter ( lambda ele: ele.find('tar') != -1 and  ele.find(file_prefix[mode])!= -1 ,  re.findall(r'"[^\"]+"', html_text)  )) # get all the quotes 

      for this_basename in basenames:
        
        this_basename = this_basename[1: -1]

        resp = requests.get(os.path.join(url, this_basename ), stream=True)
      
        with open (this_basename, "wb") as file:

          for chunk in resp.iter_content(chunk_size=8192):

            if chunk:

              file.write(chunk)
      
        # unzip compressed files 

        with tarfile.open(this_basename, 'r') as tar:

          tar.extractall(path=os.getcwd())

    except requests.exceptions.RequestException as e:
      
      print(f'An error occurred: {e}')

  def fit(self, X = None, y = None):

    return self 
  
  def transform(self, X = None, y= None):

    return X

class PreprocessSpamHam(BaseEstimator, TransformerMixin):
  
  def __init__(self, strip_header=False, lowercase= False, remove_punctuation= False,  replaceURL= False, replaceNumber= False, use_stemming= False, test_ratio=0.2, quick_run=False):
    print('Preprocess ...')

    for keyword_arg in [strip_header, lowercase, remove_punctuation , replaceURL, replaceNumber, use_stemming]:

      if type(keyword_arg)  != bool :

        raise ValueError("hyperparameters must be boolean")
      
    self.strip_header = strip_header

    self.lowercase = lowercase

    self.remove_punctuation = remove_punctuation

    self.replaceURL = replaceURL

    self.replaceNumber = replaceNumber    

    self.use_stemming = use_stemming    

    self.cvect = CountVectorizer(encoding='utf-8', lowercase=lowercase)

    self.test_ratio = test_ratio

    self.stemmer = PorterStemmer()

    self.count = 0 

    self.quick_run = quick_run

  def fit(self, X = None, y = None):

    return self

  def transform(self, dataset = None):
    
    easy_ham = os.listdir('./easy_ham')
    hard_ham = os.listdir('./hard_ham')
    spam = os.listdir('./spam')
    ham_length = len(easy_ham) + len(hard_ham)

    data = list ( map (lambda ele: ['easy_ham/' + ele, 1] , easy_ham ) ) +  list ( map (lambda ele: ['hard_ham/' + ele, 1] , hard_ham ) )  + list ( map (lambda ele: ['spam/' + ele, 0] , spam ) )
     
    # read filesname and write to DataFrame

    for i in range(len(data)):

      fname = data[i][0]

      with open ( './' + fname, 'rb') as file:

        data[i][0] = file.read()

    df = pd.DataFrame (data, columns=['WordVector', 'Safe'])

    # replace text file name with txt string file

    df['WordVector'] =   df['WordVector'].apply(self.helper_corpus)

    # build vocab using 'real' emails 
    print(f'number of instances (i.e. spam+ham emails) {df.size}')
    
    print(f'number of ham instances { ham_length }')

    self.cvect.fit(df['WordVector'].iloc[: ham_length] )

    # randomize Dataframe

    df = df.iloc[np.random.permutation(len(df))].reset_index(drop=True)

    if self.quick_run:

      df = df.iloc[:QUICK_N]
    
    sparse_ = self.cvect.transform(df['WordVector'] )

    X = sparse_.toarray()

    y = df['Safe'].copy().to_numpy()
    
    test_size = int(len(X) * self.test_ratio) 

    return X[:-test_size], X[-test_size:], y[:-test_size], y[-test_size:]
  
  def fit_predict(self, dataset = None, y = None):

    return dataset 

  def helper_corpus(self, s):
    k_index = -1 
    
    s = re.sub(rb'[\t\n\x1b.+]', b'', s) # remove tab or new lines or escape sequence

    if self.strip_header:
  
      k_index = s.find(b"Message-Id")

      if (k_index != -1):
        
        s = s[ k_index : ]
        
        # s = re.sub( rb'.*(?=Message-Id:\s*<.+>)', b'', s) # greedy  
  
    if self.lowercase:

      s = s.lower()

    if self.replaceURL:

      s = re.sub( rb'https*.+/[A-Za-z0-9-._~:/?#[\]@!$&\'()*+,;%=]*', b"URL,", s)

    if self.remove_punctuation:

      s = re.sub(rb'[.?!,;:()[\]<>{}\'"/_]', b'', s)
  
    if self.replaceNumber:
      
      s = re.sub( rb'[0-9]+', b'NUMBER' , s)

    sarray = re.split(rb'\s+', s)

    if self.use_stemming:
      
      i = 0

      while i < len(sarray):

        # for i in range(len(sarray)):  

        sarray[i] = self.stemmer.stem(sarray[i].__str__()).encode()

        if sarray[i].__len__() > (LONGEST_WORD_SIZE) :

          del sarray[i]

        else:
          
          i += 1

    s = b' '.join(sarray) 

    return s.__str__()

class TrainDataset(BaseEstimator, TransformerMixin):

  def __init__(self):

    print('Training ... ')

    self.best_cv = None

  def fit(self, dataset = None, y = None):
    print('find best classifier ... ')
    
    x_train, x_test, y_train, y_test = dataset

    best_score = float('-inf')

    current_cv = object

    for index, curr_classifier_dict in enumerate(table_of_classifiers):

      current_cv = GridSearchCV( estimator=curr_classifier_dict['estimator'](), verbose=0, return_train_score=True, scoring='f1', cv=3, refit=True, param_grid=curr_classifier_dict['param_grid'] )


      current_cv.fit(x_train, y_train)
      

      if index == 0 or best_score < current_cv.best_score_:

        self.best_cv = current_cv
        
        best_score = current_cv.best_score_ 

    print('classifier found ... done ')
    return self 
  
  def transform(self, dataset):
    
    x_train, x_test, y_train, y_test = dataset

    return x_train, x_test, y_train, y_test, self.best_cv.best_estimator_
  
class PredictDataset(BaseEstimator, TransformerMixin):
    
  def fit(self, X = None, y = None):
    
    print('Predict test set ... ')

    return self     

  def fit_predict(self, dataset = None, y = None):
    
    x_train, x_test, y_train, y_test, model  = dataset
    
    y_predictions_test = model.predict(x_test)

    conf_matrix = confusion_matrix(y_test, y_predictions_test)

    precision = precision_score(y_test, y_predictions_test)

    recall = recall_score(y_test, y_predictions_test) 

    f1 = f1_score(y_test, y_predictions_test)

    return {

      'f1_score': f1,

      'recall': recall,

      'precision': precision,

      'conf_matrix': conf_matrix,

      'conf-95': None

    }

if __name__ == "__main__":

  full_pipeline = Pipeline(
    
    [
      ('fetch_data' , FetchDataset()) ,

      ('preprocess', PreprocessSpamHam(strip_header= True, lowercase = True, remove_punctuation = True, replaceURL = True, replaceNumber =  True, use_stemming = True, quick_run=True)),
      
      ('train', TrainDataset()),
      
      ('predictions', PredictDataset())

    ]

  )

  print(full_pipeline.fit_predict(None))