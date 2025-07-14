'''

   12.

        Q. 

          Implement Batch Gradient Descent with early stopping for Softmax Regression (without using Scikit-Learn)

        A. 


'''

import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


from sklearn import datasets

EPOCHS = 220000
TEST_SET_RATIO = 0.2 
  

  
def predict(X, Y, weights, regularize= False, reg_mode = 0 ):
  """
    Softmax predicition 
  """

  target_one_hot = target_one_hot_func(Y)

  # computes scores tables 
  scores = np.dot(X, weights.T)

  scores_exp = np.exp(scores)

  # estimate probability 
  total_probability = np.sum(scores_exp, axis=1, keepdims=True)

  estimated_probability = scores_exp / total_probability

  log_estimated_probability = np.log(estimated_probability)

  # cost function   

  if regularize:
    reg_sel = None

    alpha = 0.8
    
    sum_vect = -np.sum(log_estimated_probability * target_one_hot, axis=0)

    if reg_mode == 0:
      
      reg_sel = np.sum(np.abs(weights.T), axis=0) * alpha  # lasso

    else:

      reg_sel = np.sum(np.square(weights.T), axis=0) * alpha * 0.5 # ridge
    
    cost = np.sum((sum_vect + reg_sel)) / len(X)

  else:
    
    cost = -(np.sum(log_estimated_probability * target_one_hot)/len(X)) 

  # error function 
  error = estimated_probability - target_one_hot

  # predict_class = np.argmax(estimated_probability, axis=1)

  return cost, log_estimated_probability, error

def target_one_hot_func(target, num_classes = 3):

  """
    encode target vector into one-hot vector 
  """
  num_instances = len(target)
  
  result = np.zeros( ( num_instances,  num_classes) ) 
    
  for i in range(num_instances):
    
    id = target[i]

    result[i][id] = 1

  return result

class FetchDataset(BaseEstimator,TransformerMixin):
  
  def __init__(self, test_ratio=0.2, lrate=0.006):

    self.test_ratio = test_ratio

    self.k = 3 # number of classes

    self.j = 4 # number of features 

    self.lrate = lrate

    self.theta = np.random.rand(self.k , self.j)
    
  def fit(self, X = None, y= None):
    print('fetching dataset')

    iris_dataset = datasets.load_iris()

    self.X = iris_dataset['data'][:, :4] # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

    self.Y = iris_dataset['target'] # ['setosa' 'versicolor' 'virginica']

    return self 

  def transform(self, X, y = None, **fit_params):

     # split data into test and validation 
    
    test_set_size = int(self.test_ratio * len(self.X))

    X_train, X_val, Y_train, Y_val = self.X[:-test_set_size], self.X[-test_set_size:] , self.Y[:-test_set_size], self.Y[-test_set_size:]

    return X_train, X_val, Y_train, Y_val, self.theta, self.lrate

class TrainDataset(BaseEstimator, TransformerMixin):

  def __init__(self, regularize= False, early_stop = False, reg_mode=0):
    
    self.regularize = regularize

    self.early_stop = early_stop

    self.reg_mode = reg_mode
    
  def fit(self, X = None, y = None, regularize=False):
    
    return self 
  
  def transform(self, dataset, y = None):
    print('training')

    X_train, X_val, Y_train, Y_val, theta, lrate = dataset
    
    num_instances = len(X_train)

    cost_buffer = np.zeros(shape=(EPOCHS, 3))

    min_cost = float('inf')
    
    min_epoch = 0
    
    best_weights = None
    
    max_theta = float('-inf')

    min_theta = float('inf')

    for epoch in range(EPOCHS) :

      cost_train, log_estimated_probability, error_train = predict(X_train, Y_train, theta, regularize = self.regularize, reg_mode= self.reg_mode)

      cost_val, log_estimated_probability, error_val = predict(X_val, Y_val, theta)

      grad = np.zeros(shape=theta.shape)

      for i in range(3):

        grad[i] = np.sum(error_train[:,i][:,np.newaxis] * X_train, axis =0) / num_instances 
      
      theta = theta - lrate * grad

      # axes.scatter(*theta.T[0], c=cost_train, cmap='binary', marker='o', s= 3, vmin=0, vmax=5) 
      # axes.scatter(*theta.T[1], c=cost_train, cmap='binary', marker='o', s= 3, vmin=0, vmax=5) 
      # axes.scatter(*theta.T[2], c=cost_train, cmap='binary', marker='o', s= 3, vmin=0, vmax=5) 
      # axes.scatter(*theta.T[3], c=cost_train, cmap='binary', marker='o', s= 3, vmin=0, vmax=5) 

      if np.max(theta) > max_theta:
        max_theta = np.max(theta)

      if np.min(theta) < min_theta:
        min_theta = np.min(theta)

      cost_buffer[epoch][0] = epoch

      cost_buffer[epoch][1] = cost_train 

      cost_buffer[epoch][2] = cost_val
      
      d = ( (cost_buffer[epoch ][2] - cost_buffer[epoch -2 ][2]) /  (cost_buffer[epoch ][0] - cost_buffer[epoch -2 ][0] + 0.001)   ) 

      if cost_val < min_cost :
        min_cost = cost_val
        min_epoch = epoch
        best_weights = theta.copy()
        rising_count = 0 
      elif d > 0 :
        rising_count += 1

      # early stop
      if self.early_stop and epoch >= 3000 and cost_val > min_cost and rising_count > 10000:
        break

    fig = plt.figure()

    ax = fig.add_subplot(111)

    ax.set_ylabel('Cost')

    ax.set_xlabel('Epoch')

    ax.set_xlim(0, EPOCHS)

    ax.set_ylim(0, 2)

    # ax.vlines([min_epoch], [0], [10], linewidth=1, color='green')

    ax.plot(cost_buffer[:,0][:epoch+ 1], cost_buffer[:,1][:epoch+1] , color='red', label= 'training')

    ax.plot(cost_buffer[:,0][:epoch+ 1], cost_buffer[:,2][:epoch+1] , color='blue', label= 'validation')

    ax.legend()

    print(f'best weights: {best_weights}')

    return X_train, X_val, Y_train, Y_val, {'regularize_': self.regularize, 'best_weights_': best_weights}

class MeasureModel(BaseEstimator,TransformerMixin):

  def fit(self, X=None, y=None):
    return self 
  
  def fit_predict(self, dataset = None, y = None):
    
    X_train, X_val, Y_train, Y_val, info = dataset

    plt.title(  'Regularized' if info['regularize_']  else 'Not Regularized'  )

    plt.show()

    return info


EarlyStopPipeline = Pipeline(

  [
    ('fetch_dataset', FetchDataset()), 

    ('train_dataset', TrainDataset(regularize=True, early_stop = False)), 

    ('measure_performance', MeasureModel()), 

  ]
    
)

if __name__ == "__main__":
   
  EarlyStopPipeline.fit_predict(X= None)
  