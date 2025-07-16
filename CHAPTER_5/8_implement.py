from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import numpy as np 
from matplotlib.lines import Line2D


def plot_me(ax,clf):

  xx_high_prec = np.linspace(0, 50,1000)

  yy_high_prec = np.linspace(0, 50, 1000)

  xx_high_prec_mesh, yy_high_prec_mesh = np.meshgrid(xx_high_prec, yy_high_prec)

  test_X1 =xx_high_prec_mesh.ravel()[:, np.newaxis]
  test_X2 =yy_high_prec_mesh.ravel()[:, np.newaxis]

  zz_high_prec_mesh =  clf.decision_function(   np.concatenate((test_X1,test_X2) , axis=1)   )

  # line near 0

  indices  = np.nonzero( np.logical_not((zz_high_prec_mesh >  -0.05) &  (zz_high_prec_mesh <  0.05) ) )
  xx_buffer = np.delete(xx_high_prec_mesh, indices)
  yy_buffer = np.delete(yy_high_prec_mesh, indices)
  zz_buffer = np.delete(zz_high_prec_mesh, indices)
  ax.plot(xx_buffer, yy_buffer, color='black', linewidth=0.2)

  # ax.scatter(xx_high_prec_mesh_0, yy_high_prec_mesh_0, c=zz_high_prec_mesh_0, cmap='binary')

  # line near 1

  indices  = np.nonzero( np.logical_not((zz_high_prec_mesh >  1 - 0.05) &  (zz_high_prec_mesh <  1 + 0.05) ) )
  xx_buffer = np.delete(xx_high_prec_mesh, indices)
  yy_buffer = np.delete(yy_high_prec_mesh, indices)
  zz_buffer = np.delete(zz_high_prec_mesh, indices)
  ax.plot(xx_buffer, yy_buffer, color='orange', linewidth=0.2)

  # line near -1

  indices  = np.nonzero( np.logical_not((zz_high_prec_mesh >  -1 - 0.05) &  (zz_high_prec_mesh <  -1 + 0.05) ) )
  xx_buffer = np.delete(xx_high_prec_mesh, indices)
  yy_buffer = np.delete(yy_high_prec_mesh, indices)
  zz_buffer = np.delete(zz_high_prec_mesh, indices)
  ax.plot(xx_buffer, yy_buffer, color='darkgrey', linewidth=0.2)

  plt.legend( handles=[
    
    Line2D([0], [0], color='orange', linewidth=5, linestyle='-', label='descision = 1'),

    Line2D([0], [0], color='black', linewidth=5, linestyle='-', label='descision = 0'),

    Line2D([0], [0], color='darkgrey', linewidth=5, linestyle='-', label='descision = -1')

  ])

if __name__ == '__main__':

  breat_cancer = datasets.load_breast_cancer()

  X_train, X_val , Y_train, Y_val = train_test_split( breat_cancer['data'][:,:2]  , breat_cancer['target'] , test_size=0.2)

  linear_svc_clf = Pipeline ( [
    ('scaler', StandardScaler()), 
    ('linear_svc', LinearSVC(C=1, loss='hinge'))
  ])

  xx = np.linspace(0, 50)

  yy = np.linspace(0, 50)

  fig = plt.figure()

  splot = fig.add_subplot(131)

  splot.set_xlabel('mean radius')

  splot.set_ylabel('mean texture')

  splot.set_title('LinearSVC', fontsize=8)

  linear_svc_clf.fit(X_train, Y_train) 

  splot.scatter(X_val[:,0], X_val[:,1],  c=linear_svc_clf.predict(X_val) , cmap='cool', s=10)

  plot_me(splot, linear_svc_clf)

  svc_clf  = Pipeline ( [
    ('scaler', StandardScaler()), 
    ('svc', SVC(kernel='linear', C=1))
  ])

  splot2 = fig.add_subplot(132)

  splot2.set_xlabel('mean radius')

  splot2.set_ylabel('mean texture')

  splot2.set_title('SVC', fontsize=8)

  svc_clf.fit(X_train, Y_train) 

  splot2.scatter(X_val[:,0], X_val[:,1],  c=svc_clf.predict(X_val) , cmap='cool', s=10)

  plot_me(splot2, svc_clf)

  svc_sgd_clf = Pipeline ( [
    ('scaler', StandardScaler()), 
    ('svc_svg', SGDClassifier(loss='hinge'))
  ])

  splot3 = fig.add_subplot(133)

  splot3.set_xlabel('mean radius')

  splot3.set_ylabel('mean texture')

  splot3.set_title('SGDClassifier', fontsize=8)

  svc_sgd_clf.fit(X_train, Y_train) 

  splot3.scatter(X_val[:,0], X_val[:,1],  c=svc_sgd_clf.predict(X_val) , cmap='cool', s=10)

  plot_me(splot3, svc_sgd_clf)

  fig.suptitle('Breast Cancer Short')

  plt.show()