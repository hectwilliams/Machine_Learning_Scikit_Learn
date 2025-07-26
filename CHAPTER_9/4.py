'''

    Q.
    
      What is label propagation? Why would you implement it and how?

    A.

      Label propagation is the ability to classifiy unsupervised instances which are similairy distrubuted to a small subset of supervised instances 

      Label propagation is used to classify unsupervised instances.

      Label propagation is implemented using the following guide:
      
      - Create kmeans model ( model typically fit will to dataset)

      - Find the supervised data samples , map their samples to cluster centroids with highest affinity.  These are representative data elements

      - Label unsuperivsed instances by their affinity to the representative

'''