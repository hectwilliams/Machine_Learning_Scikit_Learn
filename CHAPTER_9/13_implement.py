'''
  Continuing from ex_12 

  Reduce Olivetti faces with PCA (preserve 99% variance) and compute recontruction error for each image

  When plotting reconstructed outlier image it will try to reconstruct a normal face 

'''

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.datasets import fetch_olivetti_faces
from sklearn.mixture import  GaussianMixture
from sklearn.decomposition import PCA 
from scipy.ndimage import rotate
import matplotlib.pyplot as plt 
import numpy as np 

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

def ex13():

  faces = fetch_olivetti_faces(random_state=RANDOM_STATE)

  X, Y = faces.images, faces.target

  X_Train, X_Test, X_Val, Y_Train, Y_Test, Y_Val = split_data(X, Y)

  pca = PCA(n_components=0.99, random_state=RANDOM_STATE)

  X_Train_reduced = pca.fit_transform(X_Train)

  X_Train_reconstruct = pca.inverse_transform(X_Train_reduced)

   # damage a batch of validatiom set 
  
  gen_pic = PCG64(seed=np.random.SeedSequence(RANDOM_STATE))
  
  gen = np.random.Generator(gen_pic)
  
  indices =  np.unique( gen.integers(low= 0, high = 10, size= 5))

  for index in indices:

    shape = X_Val[index].shape
    
    pimage = X_Val[index].reshape(64,64)

    pimage = rotate(pimage, -45, reshape=False)
    
    pimage = gen.normal(0,0.1, pimage.shape) + pimage # modify pixels

    X_Val[index] = pimage.reshape (shape)

  # validation set transform and recover 

  X_Val_reduced = pca.transform(X_Val)

  X_Val_reconstruct = pca.inverse_transform(X_Val_reduced)

  recon_errors = np.zeros(len(X_Train_reconstruct))
  
  for i in range(len(X_Val_reconstruct)):

    dist_between_images = X_Val_reconstruct[i] - X_Val[i]
    
    error = np.sum(np.square(dist_between_images)) / len(X_Val_reconstruct)
    
    recon_errors[i] = error 

  fig, ax = plt.subplots()
  
  ax.scatter(range(len(recon_errors)), recon_errors, s=2)

  ax.set_xlabel('image-id', color='red')

  for id in indices:

      ax.annotate (

        text = f'{id}',

        xy = (id.astype(float), recon_errors[id]),  # point to annotate 
        
        xycoords = 'data', # data samples point coordinate system

        xytext = (10.0, 10.0 ),  # place the text at

        textcoords = 'offset pixels', 
        
        arrowprops = dict(arrowstyle= ArrowStyle.CurveB(lengthB=0.4, lengthA=0.4,widthB=4, head_width=0.2), 
                          connectionstyle="arc3,rad=0",
                          shrinkA=0.2, 
                          shrinkB=0.2 ,
                          alpha=0.2,
                          color='pink'
                          ), 
        fontsize = 6,

        color='red'

      )

  ax.set_title('Anomaly'), 
  
  ax.set_ylabel('Reconstruction Loss'), 

  plt.show()

  
if __name__ == "__main__":

  ex13()
