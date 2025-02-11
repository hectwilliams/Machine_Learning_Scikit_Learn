"""
Q. 
  If you want to build a deep sequence-to-sequence RNN, which RNN layers should have return_sequences=True?  
  What about a sequence-to-vector RNN?

A.
  A deep sequence-to-sequence RNN must have all RNN layers return sequences: time series data is passed to the output. 

Q.
  What about a sequence-to-vector RNN?
A.
  All recurrent layers except last layer: we only care about last time step when it comes to a learned vector. 
"""