'''

    Q.
    
      What is label propagation? Why would you implement it and how?

    A.

      Label propagation is the ability to classifiy unsupervised instances which are similarly distributed to a small subset of supervised instances 

      Label propagation is used to classify unsupervised instances.

      Label propagation is implemented using the following guide:
      
      - Create kmeans model (find optimal n_cluster for training set)

      - the clusters are now representaive samples. Each centroid create should map to a target

      - label unsupervised dataset to their nearest centroid cluster representative 
      

'''