'''

  Q. 
  
    When would you need to create a dynamic keras model. How do you do that? Why not make all your models dynamic?

  A. 

    All models do not need to be made dynamic because most tasks can be completed using Tensorflow's sequential, and functional model producing constructs 

    Dynamic keras model are required for models that requires complex networks that sequential and functional models cannot create.

    In order to build a dynamic keras model, one should start by subclassing and building the network (unique paths, sub-block, etc)


'''