
'''
  Train an SVM classifier on the MNIST dataset. Since SVM classifier are binary classifiers, you will need to use one-versus-the-rest.
  Tune hyperparameters using small validation sets to speed up training process. 
'''
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np 
from sklearn.model_selection import GridSearchCV

if __name__ == '__main__':

  mnist = datasets.fetch_openml('mnist_784', version=1)

  X_train, X_val , Y_train, Y_val = train_test_split(  mnist['data'], mnist['target'], test_size = 0.2)

  pipeline = Pipeline ( [
    ('scaler', StandardScaler()), 
  ])

  X_train = pipeline.fit_transform(X_train)
  X_val = pipeline.transform(X_val)

  clf = GridSearchCV(
    estimator=SVC(), 

    cv=10, 

    param_grid= {
      
      'C':  np.linspace(0.1, 5, 10).tolist(),

      'kernel': ['linear', 'rbf'],

      'gamma': [0.001, 0.1, 1, 10],
      
      'decision_function_shape' : ['ovr']

    }
  )

  clf.fit(X_train, Y_train)

  print(clf.best_estimator_)

  print(clf.best_params_)
