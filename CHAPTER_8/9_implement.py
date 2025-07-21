import time 
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

TRAIN_SIZE = 60000

def train_it(model, X, Y, X_Test, Y_Test):
  
  start = time.time()

  model.fit(X, Y)

  end = time.time()

  elapsed_time = end - start

  pred = model.predict(X_Test)

  print(f"Elapsed time: {elapsed_time:.2f} seconds")
  
  score = accuracy_score(Y_Test, pred)

  print(f"Score test set { score }", )

if __name__ == '__main__':

  mnist = datasets.fetch_openml('mnist_784', version=1)

  X, Y = StandardScaler().fit_transform(mnist['data']), mnist['target']

  X_train, X_test, Y_train, Y_test = X[:TRAIN_SIZE], X[TRAIN_SIZE:], Y[:TRAIN_SIZE], Y[TRAIN_SIZE:]

  clf = RandomForestClassifier()

  train_it(clf, X_train, Y_train, X_test, Y_test)
  
  '''
  Elapsed time: 54.48 seconds
  Score test set 0.9695
  '''

  pca = PCA (n_components=0.95)

  X_dim_reduce_train = pca.fit_transform(X_train)

  X_dim_reduce_test = pca.transform(X_test)

  clf = RandomForestClassifier()

  train_it(clf, X_dim_reduce_train, Y_train, X_dim_reduce_test, Y_test)

  '''
  Elapsed time: 207.13 seconds
  Score test set 0.9484
  '''
