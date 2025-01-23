'''
Data can be preprocessed directly when writing the data files, 
or within the tf.data pipeline, or in preprocessing layers within your model, 
or using TF Transform. Can you list a few pros and cons of each option?

Preprocessed when writing to data files 
    pros:
        - Custom management of data translation
    cons:
        - Time consuming 
        - May not be industry standard 
        - Redefine script on different deploying systems or different coding languages 

Preprocessed within tf.data pipeline 
    pros:
        - Uses tf.data API
    cons:
        - Tensor API 'hides' transformation in functional programming style 
        - Redefine tf.data pipeline on different deploying systems or different coding languages 

Preprocessing layers within model
    pros:
        - Chaining of layers
        - System design approach 
        - Add to model like any other layer object 
        - Define preprocessing operation one time (layers objects wrapped within model)

    cons:
        - Bottleneck in model is impossible to remove, without destroying pipeline 
        - Occurs during every epoch 

Preprocessing TF transform
    pros:
        - Define preprocessing operation one time 
        - End-to-End productionizing tensorflow models
    cons:
        - Requires TensorflowExtended(TFX); TFX is not bundled with Tensorflow 
'''


