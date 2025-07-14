'''

    8. 

      Q. 

        Suppose you are using Polynomial Regression. You plot the learning curves and you notice that there is alarge gap between the training error and the validation error.
        
        What is happening?
        
        What are the three ways to solve this.

      A. 

        Large gap means the model is likely overfitting  (traiining error lower than validation error)

        Couple ways to solve this:

        - reduce useless features (lasso regularization)

        - increase training data size 

        - fit model with lower polynomial degree 

        - use a largae regularization hyperparameter to reduce variance 


'''