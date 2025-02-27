"""
Q.
    Suppose you want to train a classifier, and you have plenty of unlabeled training data but only a few thousand labeled instances. How can autoencoders help? How would you proceed?

A.
    Design an Autoencoder(EncA, DecA) and train on all the training data. Constrain the 'codings dimension' to force the network to learn more aggressively. 

    Train classifier on labeled data using the EncA block as the base layer of classifier. The EncA block is analagous to giving a hint to someone lost in a maze. 

    After training classifier, the model should generalize well on all the training data.

"""