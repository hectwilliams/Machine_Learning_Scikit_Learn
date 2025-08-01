'''
Q.

  In which cases would you want to use each of the following activation functions SELU, leaky ReLU, ReLU, tanh, logistic, and softmax?

A. 

  SELU 

  - if ReLU produced vanishing currents

  - require fast covergence 

  - self normalized i/o layers 

  - using alpha dropout

  leaky ReLU

    - if ReLU produced vanishing currents

    - slow convergence is not an issue

  ReLU

    - regression model predicting output > 0

  tanh

    - regression model predicting output between -1 and 1. 

    - trinary ( three state output )

  logisitic regression

    - two state output

    - output between 0 and 1

  softmax :

    - exclusive multi-class classification 

'''