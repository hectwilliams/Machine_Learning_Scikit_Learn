"""
Q. 
How many dimensions must the input of an RNN layer have?  What does each dimension represent? 

A. 
3D array of shape [batch_size, time steps, dimensionality]
  dimensionality :
    1  - univariate
    > 1 - multivariate
Q.
What about its outputs?

A. 
Vector of shape [batch_size, n_units]
"""