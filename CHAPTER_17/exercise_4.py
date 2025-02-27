"""
Q.
    What are undercomplete and overcomplete autoencoders? 
    What is the main risk of an excessively undercomplete autoencoder? 
    What about the main risk of an overcomplete autoencoder?

A.
    Undercomplete and overcomplete are the dimensions of the codings. Undercomplete autoencoders has a lower dimensionality and vice versa for overcomplete. 
    
    The risk of excessively undercomplete autoencoders is reducing dimensionality to a point where the network can't learn much : if you lose 4 out of 5 senses can you really learn much
    
    The risk of overcomplete autoencoders is only simple patterns will be learned: to many senses will diminish learning as the network will be streteched to thin and each sense will only need to learn simple patterns as other senses will help support. 
    Learneing simple patterns about one image cannot be used to evaluate other images of varying complexity.  

     
    
"""

