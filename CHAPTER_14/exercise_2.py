'''
    Q.
    Consider a CNN composed of three convolutional layers, each with 3x3 kernels, a stride of 2, and 'same' padding. 
    The lowest layer outputs 100 feature maps, the middle one outputs 200, and the top one outputs 400.
    The input images are RGB images of 200x300x3 pixels. What are the total number of parameters. 

    A.
         (3 x 3  x 3 + 1)
    Input Layer - 
        n_parameter = 4 * 200 * 300 * 3 = 720000 Bytes
        n_parameter = 720KB
    Conv. Layer 1 - 
        Each neuron uses parameters of shape : (height_field, width_field, prev_layer_depth_size, current_layer_depth, bias_size)
            (3, 3, 3, 100, 1)
            n_parameters = (3 * 3 * 3 + 1) * 100  = 2800
    Conv. Layer 2 -
            (3, 3, 100 200, 1)
            n_parameters = (3 * 3 * 100 + 1) * 200  = 180200
    Conv. Layer 3 -
            (3, 3, 200, 400, 1)
            n_parameters = (3 * 3 * 200 + 1) * 400  = 720400

    n_parameters = 2800 + 180200 + 720400 =  903400

    Q.
    If we are using 32-bit floats, at least how much RAM will this network require when making a prediction for a single instance ?

    A.

    Input_Bytes  = (200 * 300 * 3)  * 32 = 5760000 bits 
        bits_per_input_channel = 200 * 300 * 32 = 1920000
        bits_per_input_instance = 5760000 
        bytes_per_input_instance = 720000  = 720KB

    Layer1_Bytes =
        bits_per_layer1_map = 100 * 150 * 32 = 480000 bits
        bits_feature_map = bits_per_layer1_map * 100 = 48000000 bits
        bytes_feature_map = bits_feature_map / 8 = 6000000 = 6MB  
    
    Layer2_Bytes = 
        bits_per_layer2_map = 50 * 75 * 32 = 120000 bits
        bits_feature_map2 = bits_per_layer2_map * 200 =  24000000 bits
        bits_feature_map = 3000000 = 3MB 
    
    Layer3_Bytes 
        bits_per_layer3_map = 25 x 38 x 32 = 30400 bits
        bits_feature_map3 = 12160000 bits
        bits_feature_map = 1520000 = 1.52MB
  
    
    Note: Only two consecutive layers require data to be stored in memory

    System requires at least =>  9 MB + parameter_bytes
    System requires at least =>  9 MB + (903400 * 4) = 9MB + 3.613600 MB = 12.6136 MB 

    Q.
    What about when training on mini-batch of 50 images 

    A.
    Processing 1 instance, during training each layer needs have values persist for backpropagation: parameters in all conv. layers require RAM --> 6MB + 3MB + 1.52MB = 10.5MB
    Processing_50_instance_ram = 10.5MB * 50 = 525 MB
    Input images_ram = 50 * n_parameter_input = 36 MB
    Total_parameters_ram = 3.6136 MB


 




    

'''