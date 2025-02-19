"""
Q. 
    Why do people use Encoder-Decoder RNNs rather than plain sequence-to-sequence RNNs for automatic translation.

A.
    Data going through the seq2seq network is lost at each step, losing trace of its input data as more data is fed into the network. 
    Encoder-Decoder network eliminates this short-term loss by shortening the distance between input word and translation.
"""