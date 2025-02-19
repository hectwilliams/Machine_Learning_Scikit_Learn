"""
Q. 

    What are the pros and cons of a stateful RNN versus a stateless RNN?

A.

    pros:
        - Model can learn long-term patterns
        - Previous state is preserved across batches 
        - Stateless requires more memory cell layers to help the model generalize. 

    cons:
        - Each input sequence in a batch must start exactly where the corresponding sequence in the previous left off.
        - In order to infer with datasets with varying batch sizes, create an identical stateless model and copy the weights,
"""