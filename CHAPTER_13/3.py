'''

  3. 

    During training, how can you tell that your input pipeline is the bottleneck? What can you do to fix it?

    Create a farily simple low layer neural network. Batch the data (32) and analyze the delay between epochs. 

    In order to fix bottlenecks the following will help:
    
      - Store the data as binaries will speed up data uploads.

      - If dataset is small enough the preprocessing dataset can be cached in memory to bypass preprocessing more than once. 

      
'''