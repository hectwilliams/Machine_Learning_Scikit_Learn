'''

Data can be preprocessed directly when writing the data files, or within the tf.data.pipeline, or in preprocessing layers within your model, or using TF Transform. Can you list a few pros and cons of each option?

    - Directly when writing the data files

      PROS

        - use TFRecords (compression)

        - save RAM 


      CONS 
      
        - changes requires new files to be saved 
        
        - differnet systems require additional preprocessed 
    
    - tf.data.pipeline

      PROS 

        - chain multiple processing layers 

      CONS 

        - Not portable to other systems 

        - differnet systems require additional preprocessed 

    - preprocessing layers 

      PROS

        - preprocessing Operations packed in keras model chain (hidden)

      CONS 

        - changes made to preprocessing layer may require changes to adjacent layers 

        - slow preprocessing layer will kill performance stand-alone model 

    - TF Transform 

      PROS 

        - Single preprocessing model can be shared across all systems 

      CONS 

        - If a system changes, they would have to change their system to work with the preprocessing model. 

        - Similarily, changes made for one system will affect all other systems 
        

'''