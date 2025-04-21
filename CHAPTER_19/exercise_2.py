"""
Q.
    When should you use TF Serving? What are its main features? Whare are some tools you can use to deploy it?

A. 
    Use TF Serving when you saved the computation graph(i.e. SavedModel format) entirely of tensorflow operations. Dynamic models are not included. 
    The most important feature of TF Serving is the ability to query a Docker container, which encapsulates your model, through REST and gRPC . 
    The main tools used to deploy are Docker. I recently deployed a model with an endpoint using Google Cloud AI Vertex.  
"""