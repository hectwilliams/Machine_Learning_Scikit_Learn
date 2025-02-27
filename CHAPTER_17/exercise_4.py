"""
Q.
    What are undercomplete and overcomplete autoencoders? 
    What is the main risk of an excessively undercomplete autoencoder? 
    What about the main risk of an overcomplete autoencoder?

A.
    Undercomplete and overcomplete are the dimensions of the codings. Undercomplete autoencoders has a lower dimensionality and vice versa for overcomplete. 
    The risk of excessively undercomplete autoencoders is reducing dimensionality to a point where the network can't learn much : if you lose 4 out of 5 senses can you really learn much
    Overcomplete autoencoders may produce bad 'codings' as the network won't have work hard to discover patterns: similar to an overfitting model because of its large number of dimensions.
"""
