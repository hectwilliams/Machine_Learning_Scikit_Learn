"""

Q.
Which neural network architecture could you use to classify videos?


A. 

Sequence-to-vector RNN required. 

Use the following pipeline to train a video(i.e. video frames) using the sparse_categorical_crossentropy loss function and adam optimizer :
    1D convolutional_layer(units = 10, kernel_size=4, stride=2) 
    GRU Layer (units = 10)
    GRU Layer (units = 10)
    Dense Layer (units = num_of_categories, activation="softmax")

    

"""