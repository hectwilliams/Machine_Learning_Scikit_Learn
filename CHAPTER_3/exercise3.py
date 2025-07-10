import re 
import numpy as np 
import pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

TEST_RATIO = 0.2 

class PreprocessNames(BaseEstimator, TransformerMixin):
  
  def token_array_to_custom_feature(self,name, vectorize_func):

    '''

      Takes product of distance between neareast non zero values in array. 
      
      [1 4 0 1] ------> 4

      Index 1 
        -> [1, 0] -- delta = 0

      Index 3 
        -> [3, 1] -- delta = 3
    
    '''
    name = [name] # vectorize_func accepts array only, provide single element array 
    
    token_count_table = vectorize_func.transform(name).toarray()[0]

    distance_accumulator = 0 

    tmp = [0,0]

    for i in range(len(token_count_table)):
      
      if (token_count_table[i] !=0):

          tmp[1] = tmp[0]

          tmp[0] = i
          
          distance_accumulator += (tmp[0] * tmp[1])

    return  distance_accumulator 

  def fit(self, X = None):
    
    return self 

  def transform(self, X = None):

    X['Name'] = X['Name'].apply(lambda passenger_name: re.sub(r'[.,()~-]', '', passenger_name) )

    vectorizer = CountVectorizer(encoding='utf-8', lowercase=True)

    vectorizer.fit_transform(X['Name']).toarray()
    
    X['Name'] = X['Name'].apply(self.token_array_to_custom_feature, vectorize_func=vectorizer )

    print(X['Name'] )
    assert(0) 

    return X

class Misc(BaseEstimator, TransformerMixin):
  
  def fit(self, X = None):
    
    return self 

  def transform(self, X = None):

    X['Sex'] = X['Sex'].apply(lambda sex: 0 if sex.lower().strip() == 'male' else 1 )

    return X 
  
class HandleCSV(BaseEstimator, TransformerMixin):

  """

    CSV file to Dataframe 

  """
  
  def __init__(self):
    
    print('downloading dataset  ...')

  def fit(self, X = None):

    return self
  
  def transform(self, X= None):
    
    '''
    
      features - all:

      ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
    
    '''
    
    chain_ordenc_pipeline = Pipeline (

      [
        ('ordinal_encoding', OrdinalEncoder(encoded_missing_value=-1))  
      ]

    )

    chain_std_pipeline = Pipeline (

      [
        

        ("stanardization",  StandardScaler() ) 
      ]

    )

    chain_vect_pipeline = Pipeline (

      [
        ('unique', PreprocessNames())  
      ]

    )


    chain_misc_pipeline = Pipeline (

      [
        ('miscellaneous', Misc()) , # transforms sex column
        ('imputer', SimpleImputer(strategy="median"))
      ]

    )


    preprocess_columnn_pipeline = ColumnTransformer (

      [
        # numerical column transform
        ( 'numerical', chain_std_pipeline , ['PassengerId', 'Pclass', 'SibSp', 'Parch', 'Fare' ]  ),
        
        # # ordinal column transform
        ( 'ordinal', chain_ordenc_pipeline , [ 'Embarked', 'Ticket',  'Cabin', 'Survived']  ),
        
        # # vectorize transform
        ( 'vectorize', chain_vect_pipeline , ['Name']  ),
        
        # # misc transform
        ( 'misc', chain_misc_pipeline , ['Sex', 'Age'] )

      ]

    )

    df = pd.read_csv('Titanic-Dataset.csv') 
    
    print('dataset loaded to memory  ...')

    X = preprocess_columnn_pipeline.fit_transform(df)

    X = pd.DataFrame(X, columns= [ ['PassengerId', 'Pclass', 'SibSp', 'Parch', 'Fare' ]  + [ 'Embarked', 'Ticket',  'Cabin', 'Survived']  + ['Name'] + ['Sex', 'Age'] ] ) # feature array order must match pipeline 
    
    y = X['Survived'].copy() 
    
    X = X.drop(['Survived'], axis=1)

    test_len = int(len(X) * TEST_RATIO)

    x_train, x_test, y_train, y_test = X[:-test_len] , X[-test_len:], y[:-test_len] , y[-test_len:], 

    print("data split into test and training sets ")

    return x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy().ravel(), y_test.to_numpy().ravel()


search_table = [
  {
    'estimator':  SGDClassifier,
    
    'param_grid': {
      
      'loss': ['squared_error', 'log_loss', 'hinge'],

      'alpha' : np.linspace(0.00001, 0.0001, 4), 

      'shuffle': [False, True] , 

      'n_jobs': [5],

      'learning_rate': ['optimal', 'constant'],

      'eta0': np.linspace(1e-2, 1e-3, 10), 

      'max_iter': [10000]

    },
    
    'name' : 'SGD'

  },

  {
    
    'estimator': KNeighborsClassifier,

    'param_grid':   {

      'n_neighbors' : [1,2,3,4],

      'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute' ], 

      'leaf_size': [10, 20, 30, 40]  ,

      'n_jobs' : [2,3] , 

    },

    'name' : 'kN'

  },

   {
    
    'estimator': RandomForestClassifier,

    'param_grid':   {

      'n_estimators': [3, 10, 30, 60, 90, 100],

      'max_features': [2,3,4],

      'max_depth': [5,10],

      'min_samples_split': [2,5,10],

      'min_samples_leaf': [1,2,4],

      'bootstrap': [True, False]

    },

    'name' : 'RandomForst'

  }

]

class TrainDataset(BaseEstimator, TransformerMixin):
  
  def fit(self, X):

    return self 
  
  def transform(self, dataset):
    
    print('Training ...')

    x_train, x_test, y_train, y_test = dataset

    best_ = {'best_score_': 0}

    for current_est_element in search_table:

      cv_grid_search =  GridSearchCV(

        estimator= current_est_element['estimator'](),

        param_grid= current_est_element['param_grid'],

        verbose=0,

        return_train_score = True,

        scoring = 'accuracy',

        cv = 3,

        n_jobs = 5, 

        refit=True

      )
      
      cv_grid_search.fit(x_train, y_train)

      if cv_grid_search.__dict__['best_score_'] > best_['best_score_']:

        best_['name'] = current_est_element['name']

        for key in cv_grid_search.__dict__.keys():

          if key[-1] == '_':

            best_[key] = cv_grid_search.__dict__[key]




    print(best_['best_params_'])

    print(best_['best_score_'])
    
    print('best estimator: {}'.format(best_['name']))

    print('Grid search found a model ...')

    return x_train, x_test, y_train, y_test, best_['best_estimator_']
  

class PredictDataset(BaseEstimator, TransformerMixin):

  """

    Predict Dataset
  
  """
  
  def fit(self, X = None, y = None):

    return self

  def transform(self, dataset = None):

    return dataset
  
  def fit_predict(self, dataset = None, y = None):

    print('predictions ...')
        
    x_train, x_test, y_train, y_test, model = dataset 
    
    y_predictions = model.predict(x_test)

    n_correct = sum(y_predictions == y_test)
    
    acuuracy = n_correct / len(y_predictions)

    return {

      'accuracy': round(acuuracy,4),

      'conf-95': None

    }
  
if __name__ == "__main__":

  full_pipeline = Pipeline(
    
    [

      ('preprocess_csv', HandleCSV()),

      ('training', TrainDataset()),

      ('predictions', PredictDataset())

    ]

  )

  full_pipeline.fit_predict(None)
