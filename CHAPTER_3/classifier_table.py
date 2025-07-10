
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

table_of_classifiers = [
  {
    'estimator':  SGDClassifier,
    
    'param_grid': {
      
      'loss': ['squared_error', 'log_loss', 'hinge'],

      'alpha' : np.linspace(0.00001, 0.0001, 4), 

      'shuffle': [False, True] , 

      'n_jobs': [5],

      'learning_rate': ['optimal', 'constant'],

      'eta0': np.linspace(1e-2, 1e-3, 3), 

      # 'max_iter': [1000]

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

      'n_estimators': [3, 30, 100],

      'max_features': [2,3,4],

      'max_depth': [5,10],

      'min_samples_split': [2,5,10],

      'min_samples_leaf': [1,2,4],

      'bootstrap': [True, False]

    },

    'name' : 'RandomForst'

  }

]