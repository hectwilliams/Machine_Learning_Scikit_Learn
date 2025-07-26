'''

  The classic Olivetti faces dataset

 - load dataset using sklearn.datasets.fetch_olivetti_faces 

 - split it into train, validation, and test used (use stratified samples because the dataset is small )

 - cluster images using KMeans (ensure good number of clusters)

'''

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
import numpy as np 

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

def ex10():

  faces = fetch_olivetti_faces(random_state=32)

  X, Y = faces.images, faces.target

  X_Train, X_Test, X_Val, Y_Train, Y_Test, Y_Val = split_data(X, Y)

  # find k 

  # k = find_k(X_Train)
  
  k = 62

  kmeans = KMeans(n_clusters=k, n_init=40, random_state=32)

  predictions = kmeans.fit_predict(X_Train)

  unique, counts = np.unique(predictions, return_counts=True)

  d = dict(zip(unique, counts))

  max_key = max(d, key=d.get)

  print('optimal K', k)

  plt.ion()

  for img in X_Train:

    img = img.reshape(1, 4096)

    pred = kmeans.predict(img)

    if pred == [int(max_key)]:
      plt.imshow(img.reshape((64, 64)))
      plt.axis(None)
      plt.draw()
      plt.pause(1)


if __name__ == "__main__":

  ex10()