'''
Continuing from exercise 10 using Kmean(k=62) model 

- Train a classifier (e.g. Random Forest) to predict person on each picture 

- Use kMeans as a dimension reduction tool and train on reduced set  ( performance better compared to traned classifer ?)

- Try appending features from reduced set to original features 

'''

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import  GridSearchCV
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
import numpy as np 
import math 

SEARCH_FOR_BEST_K = False
NUM_WORST_CLUSTER_REMOVE = 2
RANDOM_STATE = 32
OPTIMAL_K= 62
N_INIT = 40 


def split_data(X,Y):

  X_Test = None 
  X_Train = None 
  X_Val = None 
  Y_Test = None 
  Y_Train = None 
  Y_Val = None 
  
  split = StratifiedShuffleSplit(n_splits=3, train_size=133, random_state=32)

  for i,indices in enumerate(split.split(X, Y)):

    indices = indices[0]
    
    if i == 0:

      X_Train = X[indices].reshape(-1, 4096)
      Y_Train = Y[indices]
    
    elif i == 1:

      X_Test = X[indices].reshape(-1, 4096)
      Y_Test = Y[indices]

    elif i == 2:

      X_Val = X[indices].reshape(-1, 4096)
      Y_Val = Y[indices]
  
  return X_Train, X_Test, X_Val, Y_Train, Y_Test, Y_Val

def plot_stratified_split(Y_Train, Y_Test):

  unique_test,  counts_test = np.unique(Y_Test, return_counts=True)
  unique_train, counts_train = np.unique(Y_Train, return_counts=True)

  print(dict(zip(unique_test, counts_test)))
  print(dict(zip(unique_train, counts_train)))
  a = np.array(list(zip(unique_train, counts_train)))
  b = np.array(list(zip(unique_test, counts_test)))

  # verifyy stratified sampling 
  plt.bar(a[:,0], a[:,1], alpha=1, edgecolor='gray', facecolor='none', linewidth=1.5)
  plt.bar(b[:,0], b[:,1], alpha=1,  edgecolor='red', facecolor='none', linewidth=2.5, linestyle=':')

  plt.show()

def plot_faces(X, Y):

  print(Y)
  
  fig, axes = plt.subplots(4, 10)

  n = list(range(40))

  for i, a in enumerate(range( 0, 400, 10)):
    
    print(Y[a], a)
    r = math.floor(i / 10)

    c = i % 10
    
    axes[r, c].imshow(X[a].reshape((64, 64)))
    axes[r, c].axis('off')
    axes[r, c].set_title(i, fontsize=8)

  plt.show()

def ex11():
  
  faces = fetch_olivetti_faces(random_state=32)

  X, Y = faces.images, faces.target
  
  # plot_faces(X, Y)

  X_Train, X_Test, X_Val, Y_Train, Y_Test, Y_Val = split_data(X, Y)

  # using exercise 10 cluster k, classify validation set using RandomForest

  kmeans = KMeans(n_init=40, n_clusters=OPTIMAL_K,random_state=RANDOM_STATE) 
  
  X_Train_new = kmeans.fit_transform(X_Train)

  X_Val_new = kmeans.transform(X_Val)

  clf = RandomForestClassifier(random_state = 32)

  clf.fit(X_Train_new, Y_Train)

  accuracy_A = accuracy_score(Y_Val,  clf.predict(X_Val_new) ) # accuracy 0.8721804511278195

  print('accuracy_no_enhancements', accuracy_A)


  # remove worst cluster(s) (i.e. find average distance of centroid children for each class, and remove the lowest)

  dist = X_Train_new 

  # preprocessing training set

  dist_avg = np.sum(dist, axis= 0) / len(dist)

  worst_3_clusters_indices = np.argsort(dist_avg)[:NUM_WORST_CLUSTER_REMOVE][::-1] # those lowest affinity clusters

  X_Train_new = np.delete(dist, worst_3_clusters_indices,axis=1)
  
  # preprocessing validation set

  dist = X_Val_new 

  dist_avg = np.sum(dist, axis= 0) / len(dist)

  worst_3_clusters_indices = np.argsort(dist_avg)[:NUM_WORST_CLUSTER_REMOVE][::-1] # those lowest affinity clusters

  X_Val_new = np.delete(dist, worst_3_clusters_indices, axis=1)

  # classifer

  clf = RandomForestClassifier(random_state = 32)
  
  clf.fit(X_Train_new, Y_Train)

  accuracy_B = accuracy_score(Y_Val,  clf.predict(X_Val_new) ) # accuracy 0.8721804511278195

  print('accuracy_remove_worst_cluster \t ', accuracy_B)


  # add distance feature set (with worst cluster removed) to the original dataset ...evaluate the accuracy 

  X_Train_new_new = np.concatenate( (X_Train, X_Train_new), axis = 1)
  
  X_Val_new_new = np.concatenate( (X_Val, X_Val_new), axis = 1)
  
  clf = RandomForestClassifier(random_state = 32)

  clf.fit(X_Train_new_new,Y_Train)

  accuracy_C = accuracy_score(Y_Val,  clf.predict(X_Val_new_new) ) 

  print('accuracy_append_new_features_to_training_set  \t ', accuracy_C)


class Segmentize(BaseEstimator, TransformerMixin):
  
  def fit(self, X, y, n_clusters=5 ):

    self.n_clusters = n_clusters

    return self 
  
  def transform(self, X, y = None, **fit_params):

    pipeline = Pipeline([
      ('Serializer', Serializer()),
      ('kmeans_', K_means(n_clusters=self.n_clusters, n_init=40, random_state=32)),
      ('deserializer', DeSerializer() )
    ])
    
    
    data =  pipeline.fit_transform(X, y)
    
    print('------')
    
    return data 


class Serializer(BaseEstimator, TransformerMixin):

  def fit(self, X, y):

    return self
  
  def transform(self, X = None, y = None):

    return X.reshape(-1, 1)

  def fit_transform(self, X,Y = None):

    print('Serialize')

    return X.reshape(-1, 1)


class DeSerializer(BaseEstimator, TransformerMixin):
  def __init__(self):
    
    pass 

  def fit(self, X, y):
    
    return self
  
  def transform(self, X = None, y = None):

    return X,y

  def fit_transform(self, X,Y = None):
    
    print('Deserailize')
    
    X_new = X.reshape(133, 4096) 

    return X_new


class K_means(BaseEstimator, TransformerMixin):
  
  def __init__(self, **args):
    
    self.args = args

  def fit (self, X, y):

    return self 
  
  def transform(self, X = None, y = None):

    return X,y
  
  def fit_transform(self, X, y = None):
    
    print('Kmeans')

    kmeans = KMeans(**self.args)
    
    kmeans.fit(X) 

    segmented_samples = kmeans.cluster_centers_[kmeans.labels_] # reassign samples to nearby centroid representiatives 

    return segmented_samples



def ex11_segmentation():
  '''

    Segmentation (i.e. removing variance in pixels) was found to improve model accuracy 
  
  '''
  faces = fetch_olivetti_faces(random_state=32)

  X, Y = faces.images, faces.target
  
  # plot_faces(X, Y)

  X_Train, X_Test, X_Val, Y_Train, Y_Test, Y_Val = split_data(X, Y)

  # plot_stratified_split()

  k = 62  # from problem 10 

  print('NO SEGMENT EVAL')
  pipeline_forest_no_segment = Pipeline([
    ('random_forest', RandomForestClassifier()),
  ])
  grid_search = GridSearchCV(pipeline_forest_no_segment, cv=3, param_grid=[{'random_forest__random_state': [32]}], refit=True)
  grid_search.fit(X_Train, Y_Train)
  print('grid search CV best score \t ', grid_search.best_score_) 
  accuracy = accuracy_score(Y_Val, grid_search.predict(X_Val)) # accuracy 0.8721804511278195
  print('validate set accuracy', accuracy)

  print('SEGMENT EVAL')

  # reduce dimension 
  pipeline_forest_segment = Pipeline([
      ('segmentize', Segmentize()),
      ('random_forest', RandomForestClassifier(random_state=32)),
    ])

  best_score = 0 

  best_k = 7 
  
  if SEARCH_FOR_BEST_K:

    for k in range(2, 15):
      
      print(f'current k = {k}')

      pipeline_forest_segment.fit(X_Train, Y_Train, segmentize__n_clusters = k)
      
      accuracy = accuracy_score(Y_Val, pipeline_forest_segment.predict(X_Val))

      if accuracy > best_score:
        
        best_score = accuracy
        
        best_k = k
    
  pipeline_forest_segment.fit(X_Train, Y_Train, segmentize__n_clusters = best_k)
    
  accuracy = accuracy_score(Y_Val, pipeline_forest_segment.predict(X_Val))

  print('validate set accuracy', accuracy)

if __name__ == "__main__":

  ex11()