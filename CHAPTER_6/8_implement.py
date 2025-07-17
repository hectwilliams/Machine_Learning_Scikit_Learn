
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
from matplotlib.patches import Circle
from sklearn.model_selection import ShuffleSplit
from scipy.stats import mode 
import joblib


# running exercise 7 prior to running this script 

info = joblib.load("info.pkl")
params = info['best_params']
X_train = info['X_train']
X_val = info['X_val']
Y_val = info['Y_val']
Y_train = info['Y_train']

# split training set int 1000 subsets of size 100 ( 10% )
rand_splitter = ShuffleSplit(n_splits=1000, random_state=42, train_size=0.1)

evals = np.zeros(shape=(1000))

majority_vote_pred = np.zeros(shape=(len(X_val)))

dtrees = [DecisionTreeClassifier(**params) for _ in range(1000)]

# train all subsets 1000 and measure accuracy on the 'one and only' test set 

for i, (train_indices, test_indices) in enumerate(rand_splitter.split(X_train)):
  
  # dtree_clf = DecisionTreeClassifier(**params)

  X_train_subset = X_train[train_indices]

  Y_train_subset = Y_train[train_indices]

  dtrees[i].fit(X_train_subset, Y_train_subset)

  evals[i] = accuracy_score(Y_val, dtrees[i].predict(X_val))

for i in range(len(majority_vote_pred)):
  
  test_instance = X_val[i] # [-0.23472474 -0.04066031]

  test_instance = test_instance[ np.newaxis,: ] # [[-0.23472474 -0.04066031]]

  curr_predictions = list(map(lambda model: model.predict(test_instance)[0], dtrees))

  majority_vote_pred[i] = mode(curr_predictions).mode
  
accuracy = accuracy_score(majority_vote_pred, Y_val )

print(f'1000 Decision-split evals {evals}')

print(f' Majority-vote predictor {accuracy}')
