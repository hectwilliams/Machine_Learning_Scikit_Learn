"""
    Q. 
        What are the different ways TFLite reduces a model's size to make it run on a mobile or embedded device?

    A.
        Compresses it to a much lighter format based on FlatBuffers by doing the following:
            - reduce the model size, to shorten download time and reduce RAM usage 
            - reduce the amount of computations needed for each prediction, to reduce latency, battery usage, and heating
            - adapt the model to device specific contraints 
"""