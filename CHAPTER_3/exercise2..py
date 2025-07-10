import pprint 
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.base import clone 
from sklearn.datasets import fetch_openml
import numpy as np 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier
from scipy.ndimage import shift

TEST_RATIO = 0.20
DEBUG_COUNT = 5000

def plot_images(images, n=5):

  fig, axes = plt.subplots(nrows=n , ncols=n)
  index = 0
  
  for r in range(n):

    for c in range(n):

      ax = axes[r,c]

      index = r * n + c 

      img = images[index].reshape(28, 28)

      ax.imshow( img , cmap='binary') 

      ax.axis('off')

  plt.show()

class DownloadMnist(BaseEstimator, TransformerMixin):
  '''

    Grab mnist dataset and split into dataset-train and dataset-test

  '''
  def fit(self, X = None, fast_eval = False):
    print('downloading dataset  ...')

    self.fast_eval = fast_eval
    return self 

  def transform(self, X = None, fast_eval = False):

    mnist = fetch_openml('mnist_784', version=1)

    X, y = mnist['data'], mnist['target']

    print(f'full weight {len(X)} ')

    y = y.astype(np.uint8) 
    
    y = (y == 5) #  5 or not 5

    test_set_size = int(len(y) * TEST_RATIO)

    x_train, x_test, y_train, y_test  = X[:-test_set_size].to_numpy(), X[-test_set_size:].to_numpy(), y[:-test_set_size].to_numpy(), y[-test_set_size:].to_numpy()

    if (self.fast_eval) :
      x_train = x_train[:DEBUG_COUNT]
      y_train  =  y_train[:DEBUG_COUNT]

    return x_train, x_test, y_train, y_test

class PrepareDataset(BaseEstimator, TransformerMixin):
  '''

    Data Augmentation layer, for each image, shift it 4 ways and add to training set

  '''

  def fit(self, X = None):
    print('dataset augmentation ...')
    
    return self 

  def transform(self, dataset = None):
    
    shift_size = 5

    x_train, x_test, y_train, y_test = dataset 

    augmented_images_index = 0

    augmented_images = [ np.zeros(shape=(784,))   for i in range(len(x_train) * 4 ) ]
    augmented_images_class = [ False  for i in range(len(y_train) * 4 )  ]

    for img in x_train:
      
      img = img.reshape(28,28)
      
      # left shift 
      left_shift_img = img.copy()
      left_shift_img = shift(left_shift_img, [0,-shift_size])  

      # right shift
      right_shift_img = img.copy()
      right_shift_img = shift(right_shift_img, [0,shift_size])  

      # up shift
      up_shift_img = img.copy()
      up_shift_img = shift(up_shift_img, [-shift_size,0])  
      
      # down shift
      down_shift_img = img.copy()
      down_shift_img = shift(down_shift_img, [shift_size,0])  
      
      augmented_images[augmented_images_index*4    ] = left_shift_img.reshape(784)
      augmented_images[augmented_images_index*4 + 1] = right_shift_img.reshape(784)
      augmented_images[augmented_images_index*4 + 2] = up_shift_img.reshape(784)
      augmented_images[augmented_images_index*4 + 3] = down_shift_img.reshape(784)

      augmented_images_class[augmented_images_index*4    ] = y_train[augmented_images_index]
      augmented_images_class[augmented_images_index*4 + 1] = y_train[augmented_images_index]
      augmented_images_class[augmented_images_index*4 + 2] = y_train[augmented_images_index]
      augmented_images_class[augmented_images_index*4 + 3] = y_train[augmented_images_index]

      augmented_images_index  += 1
     
    x_train = np.vstack((x_train, np.array(augmented_images))) # arrays are greater than 1-D
    y_train = np.hstack((y_train, np.array(augmented_images_class))) # arrays are 1-D

    return x_train, x_test, y_train, y_test 
 
cv_grid_search_kneighborclassifier = GridSearchCV(
            
    estimator = KNeighborsClassifier(),

    param_grid = [

      { 
        'n_neighbors' : np.arange(6, 7).tolist() , 

        'weights': ['distance'], 

        'n_jobs' :  [4]

      },

    ],

    return_train_score = True,

    scoring = 'accuracy',

    cv = 5,

    n_jobs = 5, 

    refit=True

)

class TrainDataset(BaseEstimator, TransformerMixin):
  def fit(self, X = None):
    print('training ...')
    
    return self 

  def transform(self, dataset = None):

    x_train, x_test, y_train, y_test = dataset 

    cv_grid_search_kneighborclassifier.fit(x_train, y_train)

    print(f' Optimal K_neighbors parameters:   {cv_grid_search_kneighborclassifier.best_params_}')
    print(f' Grid Search Cross-Validation best score (cv= 5):   {cv_grid_search_kneighborclassifier.best_score_}')

    return  x_train, x_test, y_train, y_test , cv_grid_search_kneighborclassifier.best_estimator_


class PredictDataset(BaseEstimator, TransformerMixin):

  def fit (self):
    return self; 

  def transform(self, dataset):
    return dataset 

  def fit_predict(self, dataset, y = None):
    print('predictions ...')
        
    x_train, x_test, y_train, y_test, model = dataset 
    
    y_predictions = model.predict(x_test)

    n_correct = sum(y_predictions == y_test)
    
    acuuracy = n_correct / len(y_predictions)

    return {

      'accuracy': round(acuuracy,4),

      'conf-95': None

    }

if __name__ == '__main__':

  full_pipeline = Pipeline(

      steps = [

          ('download_mnist', DownloadMnist()),
          
          ("prepare_dataset", PrepareDataset()),

          ("fit_training", TrainDataset()), 

          ("predictions", PredictDataset())
          
      ]

  )

result = full_pipeline.fit_predict(None, download_mnist__fast_eval= True)

print(result)