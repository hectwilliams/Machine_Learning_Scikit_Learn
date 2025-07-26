'''
  Train Gaussian mixture model on Olivetti faces dataset 

  - generate new faces 

  - moodify images and show they are anomalies 

  - detect anomalies 
'''

from sklearn.datasets import fetch_olivetti_faces
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.mixture import  GaussianMixture
from sklearn.decomposition import PCA 
from scipy.ndimage import rotate

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

def ex12():
  
  faces = fetch_olivetti_faces(random_state=32)

  X, Y = faces.images, faces.target
  
  X_Train, X_Test, X_Val, Y_Train, Y_Test, Y_Val = split_data(X, Y)

  # modify first validatiom image rotate first image 
  
  shape = X_Val[0].shape
  
  pimage = X_Val[0].reshape(64,64)

  image_rotate = rotate(pimage, 45, reshape=False)

  X_Val[0] = image_rotate.reshape (shape)
  
  # PCA  99% variance 

  pca = PCA(n_components=0.995)

  X_Train_reduced = pca.fit_transform(X_Train)

  X_Val_reduced = pca.transform(X_Val)

  gm = GaussianMixture(n_components=X_Train_reduced.shape[-1], n_init=N_INIT)

  gm.fit(X_Train_reduced)

  # generative_img_samples(pca, gm)

  '''
    log probability density function estimate location of samoles (i.e. image)
    
    higher score -  high probabiltiy of being found

    low score  -  low probabiltiy of being found ( an outlier )
  '''
  scores = -gm.score_samples(X_Val_reduced) # invert
  scores = scores/ np.max(scores) # normalize (low score - outlier, high score - dsitributes well with data )

  plt.title('Log PDF') # estimates log pdf for images location
  plt.figure(1)
  plt.axis(None)
  plt.vlines(range(133), np.zeros((133,)), scores)
  plt.show() 


  if __name__ == "__main__":

    ex12()

