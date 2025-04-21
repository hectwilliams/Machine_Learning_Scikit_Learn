"""
    Q.  
        How do you deploy a model across multiple TF Serving instances?

    A. 
        Distribute many Docker (or other TF Serving applications) containers across multiple servers and load-balance queries. 

        Use tools such as Kubernetes to manage deployment and managing of many TF Serving containers.
        
        The big dogs Amazon AWS , IBM cloud , Google Cloud, have tools that automate this as well for the right price.

"""