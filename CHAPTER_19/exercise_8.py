"""
    Q.
        When training a model across multiple servers, what distribution strategies can you use. How do you choose which on to use?
    
    A.
        Training a model across multiple servers is called a cluster.

        Tensorflow provides API to handle distribution strategies:
            - train across all available GPUS on a single machine using mirroed strategy 
            - train across subset of GPUs
            - data parrallism with centralized parameters
            - asynchronous updates
            - synchronous updates 
            ...

        Choice depends on model performance. Heavy computations (large matrix mults) run faster on powerful GPUs 

        Choice depends on Scarcity of GPUs.
         
        Choice depends on the  performance on TPUs.

        Choice depends on CPU performance 

        Choice depends on CPU and GPU performance

        Choice depends on training time(maybe distribute across multiple GPUs)


"""