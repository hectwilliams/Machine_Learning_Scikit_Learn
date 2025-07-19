'''
  
  1) Train mnist on soft or hard ensemble on voting classifier ( create your own)

  2) Using classifiers in previous step create a blender model and test it on test set

'''
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import numpy as np 
from scipy.stats import mode 
from sklearn.metrics import accuracy_score

VOTING = 'soft'
REDUCED_TRAINING_SIZE = 0 # replace with value >0 and less than 20000 for quick test 

def majority_vote (classifiers, X, Y, name='',  voting= 'hard'):
  '''
    trains and print majority-vote scores and scores of each independent classifier
  '''
  
  pred_ = np.zeros(shape=(len(Y)))
  
  if voting == 'hard' :

    clfs_predictions = [this_classifier.predict(X) for this_classifier in classifiers]

    pred_votes = np.concatenate( [current_clf[:, None] for current_clf in clfs_predictions], axis = 1 )

    pred_ = np.apply_along_axis(lambda votes: mode(votes).mode , axis=1, arr=pred_votes)

  else:
    
    clfs_predictions = np.array([this_classifier.predict_proba(X) for this_classifier in classifiers]) # (3, 10000, 10)
    
    pred_votes = np.concatenate( [current_clf[:, None] for current_clf in clfs_predictions], axis = 1 ) # (10000,3, 10)
    
    avg_probabilities = np.array([ np.sum(current_votes_matrix, axis=0 )/len(clfs_predictions) for current_votes_matrix in pred_votes])  # (10000, 10)

    pred_ = np.argmax(avg_probabilities, axis=1) # (10000, 1)

  scores_val = dict ( scores =  [ accuracy_score(Y, pred_) , "majority-vote" ] )

  for i, this_clf in enumerate(classifiers):
    
    scores_val[i] = [accuracy_score(Y,  np.argmax( clfs_predictions[i], axis=1)   ) ,  type(this_clf).__name__]

  scores_val['name'] = name

  print(scores_val)

mnist = datasets.fetch_openml('mnist_784', version=1)

X_train, X_val , Y_train, Y_val = train_test_split(  mnist['data'], mnist['target'], test_size = 0.28)

# standardize
X = StandardScaler().fit_transform(mnist['data'])

Y = mnist['target'].astype(int)

# split dataset 
X_train, X_val, X_test, Y_train, Y_val, Y_test = X[0:REDUCED_TRAINING_SIZE], X[50000:60000], X[60000:70000], Y[0:REDUCED_TRAINING_SIZE], Y[50000:60000], Y[60000:70000]

# classifiers 
rand_forest_clf  = RandomForestClassifier()

extra_trees_clf  = ExtraTreesClassifier() 

svm_clf = SVC( probability=(VOTING == 'soft') )

# train classifiers on training data 
rand_forest_clf.fit(X_train, Y_train)

extra_trees_clf.fit(X_train, Y_train)

svm_clf.fit(X_train, Y_train)

bag_of_classifiers = (rand_forest_clf, extra_trees_clf, svm_clf)

majority_vote( (bag_of_classifiers), X_val, Y_val, name='validation', voting=  VOTING)

majority_vote( (bag_of_classifiers), X_test, Y_test, name='test' , voting= VOTING)

# * blender stitch *

rpred = np.argmax(rand_forest_clf.predict_proba(X_val),  axis=1)

epred = np.argmax(extra_trees_clf.predict_proba(X_val), axis=1 )

spred = np.argmax(svm_clf.predict_proba(X_val) , axis=1)

# create blender training set 
blender_train_X_val = np.concatenate((rpred[:, None], epred[:, None], spred[:, None]), axis = 1)

blender_classifier = SVC(probability=(VOTING == 'soft'))

blender_classifier.fit(blender_train_X_val, Y_val)


# create blender test sset 
rpred = np.argmax(rand_forest_clf.predict_proba(X_test),  axis=1)

epred = np.argmax(extra_trees_clf.predict_proba(X_test), axis=1 )

spred = np.argmax(svm_clf.predict_proba(X_test) , axis=1)

# evaludate test set 
blender_test_X_test = np.concatenate((rpred[:, None], epred[:, None], spred[:, None]), axis = 1)

blender_y_pred = np.argmax(blender_classifier.predict_proba(blender_test_X_test), axis=1)

print(f'ensemble score:  {accuracy_score(Y_test, blender_y_pred)}')


'''

Most Recent Run Train_Size=30000

{'scores': [0.9746, 'majority-vote'], 0: [0.969, 'RandomForestClassifier'], 1: [0.9715, 'ExtraTreesClassifier'], 2: [0.9652, 'SVC'], 'name': 'validation'}
{'scores': [0.9699, 'majority-vote'], 0: [0.966, 'RandomForestClassifier'], 1: [0.9676, 'ExtraTreesClassifier'], 2: [0.9591, 'SVC'], 'name': 'test'}
ensemble score:  0.9624

'''