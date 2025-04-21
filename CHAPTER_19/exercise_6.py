"""
    Q.
        What is quantization-aware training, and why would you need it?

    A.
        Reduce the model size, by quantizing the model weights down to fixed point, 8-bit integers. 

        Converting 32 floats to fixed point to 8 bit integers for transmission. At runtime the quantized values get converted back with small error.

        This method reduces model size, and it's much faster to download and store. 


"""