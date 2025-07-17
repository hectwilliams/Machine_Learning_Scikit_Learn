
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

def exercise_7 () :

  X, y = make_moons(int(1e4), noise=0.4)
  
  X_train, X_val , Y_train, Y_val = train_test_split(X, y, test_size=0.2)


  clf_search_cv = GridSearchCV(
    
    estimator=DecisionTreeClassifier(),

    cv = 3, 

    refit= True,
    
    param_grid= {
      
      'max_leaf_nodes' : np.arange(2, 10),

      'max_depth': [2, 4, 7, 10, 12, 15],

      'min_samples_leaf': [5, 6, 7,8, 10] , 

      'criterion': ['gini', 'entropy']

    }
  )

  clf_search_cv.fit(X_train, Y_train)

  info = {
    
    'best_params': clf_search_cv.best_params_,
    
    'best_estimator': clf_search_cv.best_estimator_, 

    'best_score': clf_search_cv.best_score_,

    'X_train': X_train, 

    'Y_train': Y_train, 

    'X_val': X_val, 

    'Y_val': Y_val

  }

  model = clf_search_cv.best_estimator_
  
  Y_pred = model.predict(X_val)

  x = np.linspace(-3, 4, 100)
  y = np.linspace(-3, 4, 100)
  
  xx, yy = np.meshgrid(x, y)

  xx_test = xx.ravel()[:, np.newaxis]

  yy_test = yy.ravel()[:, np.newaxis]

  zz_test = np.concatenate((xx_test,yy_test) , axis=1) 
  zz_pred =  model.predict(zz_test)

  plt.scatter(zz_test[:,0],zz_test[:,1],  c=zz_pred.ravel() * 10, cmap='summer')
  plt.scatter(X_val[:,0],X_val[:,1],  c=Y_pred.ravel() * 10, cmap='winter',alpha=0.3)

  plt.legend( handles=[
  
    mpatches.Patch(color= cm.get_cmap('summer')(0.0) , label='0') ,
    mpatches.Patch(color= cm.get_cmap('summer')(1.0) , label='1') ,
    Line2D(range(1), range(1), color="white", marker='o', markerfacecolor=cm.get_cmap('winter')(0.0), label='0' ),
    Line2D(range(1), range(1), color="white", marker='o', markerfacecolor=cm.get_cmap('winter')(1.0) , label='1' )

  ])

  with open('info.pkl', 'wb') as file:

    pickle.dump(info, file)

  plt.show()
