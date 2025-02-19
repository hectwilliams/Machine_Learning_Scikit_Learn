"""
Q. 
    What is the most important layer in the Transformer architecture? What is the purpose?

A. 
    The most important layer is Multi-Head Attention layer. It's purpose is to pay attention to relevant words in a sentence. 
    
    Encoder:
        A  word(i.e. vectorized) will encode importance or relationship to all words in a sentence (including itself). 
    
    Decoder:
        Pays attention to words in the input sentence when outputting a word translation. A certain word will be focused on during a word translation. 
        Ex. During a French translators next output of the word 'lait', the Transformer will focus on the input word 'milk' 
"""
