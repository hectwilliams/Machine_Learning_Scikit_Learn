"""
    Q.
        How do you tie weights in stacked autoencoder? What is the point of doing so?
    
    A.
        Encoder and deocder share encoder weights. This helps speed up training and limits risk of overfitting(constraint). 
"""