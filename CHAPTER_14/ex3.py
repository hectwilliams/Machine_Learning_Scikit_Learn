Q.

  If GPU runs out of memory while training a CNN, what are five things you would try to solve the problem?

A

  - use data type with smaller bit length

  - delete previous layer data once new layer input is computed 

  - have CNN train objects on smaller inputs and then use fully convolutional to images of larger sizes

  - reduce kernel field size

  - use stride of 2 to reduce input size ( max pooling )

  - reduce number of feature map channels
