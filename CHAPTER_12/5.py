'''

    Q.

      A custom loss function can be defined by writing a function or by subclassing the keras.losses.Loss class. When would you use each option?

    A.

      If hyperparameters need to be saved or more granularity is required for the function (e.g. multi-stage), subclassing is required. 
      
      Custom loss function are used when persistence does not matter. 

'''