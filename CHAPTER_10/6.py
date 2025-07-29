'''

  Q. 

    Suppose you have an MLP composed of one input layer with 10 passthrough neurons, 
    followed by one hidden layer with 50 artificial neurons, 
    and finally one output layer with 3 artificial neurons. All artificial neurons use the ReLU activation function.

    Q.1
      
      What is shape of input matrix?

    A.1 

      N x 10 


    Q.2
      
      What are the shapes of the hidden layers weight vector Wh and its bias vector bh?

    A.2 

      W_h = 10 x 50 

      b_h = 1 x 50

      
    Q.3
      
      What are the shapes of the hidden layers weight vector Wo and its bias vector bo?
    
    A.3

      W_o = 50 x 3

      b_o = 1 x 3 

    
    Q.4

      What is the shape of the networks output Matrix Y 

    A.4

      Y = 1 x 3 

    Q.5 

      Write the equation that computes networks output Matrix Y as function of X, W_h , b_h, W_o , b_o 

    A.5

      Y = (X.dot(W_h) + b_h).dot(W_o) + b_o 

'''