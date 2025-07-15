import numpy as np 
import matplotlib.pyplot as plt 

from  sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from scipy.special import expit, logit

def learning_curves(model, X, y):
  '''
    returns validation and training learning curves 
  '''

  size = len(X)

  X_train, X_val , y_train, y_val = train_test_split(X, y, test_size=0.2)

  train_errors, val_errors = [], [] 

  for m in range (1, len(X_train)):

    batch_x_train = X_train[:m]

    batch_y_train = y_train[:m]

    model.fit( batch_x_train, batch_y_train)

    # model predicts on subset of instance
    y_train_predict = model.predict(batch_x_train) 

    # model predicts using entire validation set
    y_val_predict = model.predict(X_val)

    train_errors.append(mean_squared_error(batch_y_train, y_train_predict))

    val_errors.append(mean_squared_error(y_val, y_val_predict))

  return np.sqrt(train_errors), np.sqrt(val_errors), np.arange(len(train_errors))

def poly_routine_run():
     

  m = 100 

  X = 6 * np.random.rand(m, 1 ) -3

  y = X**2 + X + np.random.randn(m, 1)


  train_set_sizes = np.arange(len(X))


  # use pipelines 

  regression_pipeline = Pipeline(

    [
      ('linear_regression', LinearRegression())
    ]
      
  )

  poly_regression_pipeline = Pipeline(

    [
      ('feature_2nd_degree', PolynomialFeatures(degree=2, include_bias=False)), 
    ]
      
  )

  super_poly_regression_pipeline = Pipeline(

    [
      ('feature_60nd_degree', PolynomialFeatures(degree=90, include_bias=False)), 
    ]
      
  )

  pipeline_dict = {
        
        'reg': LinearRegression(),

        'poly_pipeline': [

          None, 
          poly_regression_pipeline,
          super_poly_regression_pipeline
        ]
  }
  pipelines_ = [regression_pipeline, poly_regression_pipeline, super_poly_regression_pipeline]

  fig, ax = plt.subplots(nrows=len(pipelines_),ncols=2)

  for i in range(len(pipelines_)):

    # X_new = curr_pipeline.fit_transform(X)
    poly_pipe = pipeline_dict['poly_pipeline'][i]
    
    # Augment feature set 
    X_new = X if not poly_pipe else poly_pipe.fit_transform(X)

    # learn from feature set
    pipeline_dict['reg'].fit(X_new, y)

    ax[i, 0].scatter(X, y, label='true', s=1)
    sorted_rv  = sorted((zip(X, pipeline_dict['reg'].predict(X_new))) , key= lambda e: e[0]) # X is random variables, sort by X
    ax[i, 0].plot( *list(zip(*sorted_rv)), label='predict', color='orange')
    ax[i, 0].legend()

    train_errors, validation_errors, train_sizes = learning_curves(  pipeline_dict['reg'],  X_new, y)
    ax[i, 1].plot( train_sizes,  train_errors, label='train' )
    ax[i, 1].plot( train_sizes,  validation_errors, label='validation' )
    ax[i, 1].legend()
    ax[i, 1].set_ylabel('RSME')

    if i == 0:
      ax[i, 0].set_title('Fit')
      ax[i, 1].set_title('Underfitting', fontsize= 8)

    elif i == 2:
      ax[i, 1].set_xlabel('Training_set_size')
      ax[i, 1].set_title('Overfitting', fontsize= 8)

    elif i == 1:
          ax[i, 1].set_title('Optimal Fit', fontsize= 8)

  plt.show()

def logistics():
   
   fig, axes = plt.subplots(2)

   t = np.linspace(-5, 5, 1000)

   y = expit(t)
   
   axes[0].set_title('logistic')

   axes[0].plot(t, y)
   
   axes[0].set_ylim(0, 1)
   
   y = logit(t)
      
   axes[1].set_title('logit')

   axes[1].plot(t, y)
   
   axes[1].set_xlim(0, 1)
   
   axes[1].set_ylim(-5, 5)
   
   plt.show()

def logistics2():
   
   fig, axes = plt.subplots(ncols=2)

   p = np.linspace(0, 1, 1000)

   y = -np.log(p)
   
   axes[0].set_title('Positive Class Estimator', fontsize=8)

   axes[0].plot(p, y)
   
   axes[0].set_xlabel(' probability  ', fontsize=8)
   
   axes[0].set_ylabel('Cost', fontsize=8)
   
   axes[0].set_xlim(0.5, 1)
   
   axes[0].set_xlim(0,1)
   
   axes[0].set_yticks([])  

   y = -np.log(1 - p)
      
   axes[1].set_title('Negative Class Estimator', fontsize=8)

   axes[1].plot(p, y)

   axes[1].set_xlabel(' probability  ', fontsize=8)
   
   axes[1].set_ylabel('Cost', fontsize=8)

   axes[1].set_xlim(0, 0.5)

   axes[1].set_xlim(0,1)
   
   axes[1].set_yticks([])  
   
   plt.show()

def exercises():
  
  '''
    
    1.
      
    Q.

      Which Linear Regression training algorithm can you use if you have a training set with million of features 

    A.

      Gradient Descent preferred instread on closed form solution

    2. 

      Q.

        Suppose the features in your training set have different scales. What algorithm might suffer from this, and how? What can you do about it?

      A.

        Gradient descent suffers a likely long wait time to find global minimum. Ensuring feature data have similar scale will speed up convergence

    3.

    
    Q.

      Can Gradient Descent get stick in a local minimum when training a Logistic Regression model 

    A.

      If the learning rate is large, it will overshoot the optimal path towards convergence 


    4.

    Q. 
    
      Do all Gradient Descent algorithms lead to the same model, provide you leftthem run long enough 

    A.

      Provide a optimal learning schedule all gradient descient algorithms will eventually converage, though wait times will vary

    5.
      Q.

      Suppose you use Batch Gradient Descent and you plot the validation error at ever epoch. If you notice the validation error consistent goes up, what is likely going on. How can you fix this?

      A.

      Model is suffering from large bias ( low variance) 

      Consider adding more features, and/or more training data.


    6.

      q. 

      Is it a good idea to stop Mini-batch Gradient Descent immediately when the validation error goes up 


      A. 

      I would be concerned as mini-batch learning steps should converage well compared to stochastic algorithm. 
      If the validation error grows for many tranining steps and does not fix to a converage state then I would stop the training process 

    7.

    Q.

    Which Gradient Decent Algorithm will reach the vicinity of the optimal solution the fatest? Which will actually converage. How can you make the others converage as well?

    A. 

    Mini-Batch Gradient Descent is the fastest

    Batch Gradient Descent is likely to converage (provided a good training dataset)

    A good learning schedule will help the alogirthm converage better 

    
    8. 

      Q. 

        Suppose you are using Polynomial Regression. You plot the learning curves and you notice that there is alarge gap between the training error and the validation error.
        What is happening?
        What are the three ways to solve this.

      A. 

        Large gap means the model is likely overfitting  (traiining error lower than validation error)

        Couple ways to solve this:

        - reduce useless features 

        - increase training data size 

        - fit model lower polynomial degree 

        - use a largae regularization hyperparameter to reduce variance 

    9.

      Q.

      Suppose you are using Ridge Regression and you notice that the training error
      and the validation error are almost equal and fairly high. 
      Would you say that the model suffers from high bias or high variance?
      Should you increase the regularization hyperparameters (alpha-param) or reduce

      A.
      
      Model is undefitting, sufferring from high bias 

      Lowering the regularization is required to lower bias 

    
    10. 
      
      Q.

      Why would you want to use

      a. Ridge Regression instead of plain Linear Regression

      b. Lasso instread of Ridge Regression 

      c. Elastic Net instread of Lasso 

      A. 

        a.

          - smaller model parameters 

          - reduce model variance 

        b. 

          - removes least important features 

        c.
          
          -  if number of features is large 

          - if features are strongly correlated. 

    11.

      Q. 

        Suppose you want to classify pictures as outdoor/indoor and daytime/nighttime. 

        Should you implments two logistic regression classifier or one softmax regression classifier 

      A.

         Use two logistic regression classifiers 

    12.

      Q. 

        Implement Batch Gradient Descent with early stopping for Softmax Regression (without using Scikit-Learn)

      A. 

        ... 
  '''
  
from sklearn import datasets

EPOCHS = 20000
TEST_SET_RATIO = 0.2 
REGULARIZATION_HYPERPARAMETER = 1

  
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

    sum_vect = -np.sum(log_estimated_probability * target_one_hot, axis=0)

    if reg_mode == 0:
      
      reg_sel = np.sum(np.abs(weights.T), axis=0) * REGULARIZATION_HYPERPARAMETER  # lasso

    else:

      reg_sel = np.sum(np.square(weights.T), axis=0) * REGULARIZATION_HYPERPARAMETER * 0.5 # ridge
    
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

  def regu_lasso_grad(self, val):

    if val < 0:

      return -1 
    
    elif val  == 0: 

      return 0
    
    elif val > 0:

      return 1
   
  def regu_ridge_grad(self, val):
    return val 
 
  def handle_reg(self, theta, regu_partial_matrix ):
    regu_rate = 0.8

    for i in range(3):

      for k in range(4):
        
        if self.reg_mode == 0:

          regu_partial_matrix[i][k] = self.regu_lasso_grad(theta[i][k] ) * REGULARIZATION_HYPERPARAMETER
        
        elif self.reg_mode == 1:

          regu_partial_matrix[i][k] = self.regu_ridge_grad(theta[i][k] ) * REGULARIZATION_HYPERPARAMETER
  
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

      regu = np.zeros(shape=theta.shape)
      
      if self.regularize:
        
        self.handle_reg(theta, regu)

      for i in range(3):
        
        grad[i] = ( np.sum(error_train[:,i][:,np.newaxis] * X_train, axis =0) / num_instances )
      
      theta = theta - (lrate * (grad + regu) )

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

    ('train_dataset', TrainDataset(regularize=True, early_stop = False,reg_mode=1)), 

    ('measure_performance', MeasureModel()), 

  ]
    
)

if __name__ == "__main__":
   
  EarlyStopPipeline.fit_predict(X= None)