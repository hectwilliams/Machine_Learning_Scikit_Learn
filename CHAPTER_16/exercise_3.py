"""
Q. How can you deal with variable-length input sequences. What about variable length output sequences. 

A. Input sequences can be bucketized and padded. Batches can be different sizes, as long as the batche elements have the same sequence length. 
   Output sequences masking can be used. 
"""