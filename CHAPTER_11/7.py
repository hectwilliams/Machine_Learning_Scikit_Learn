'''

  7. 

    Q.

      Does dropout slow down training? 
      
      Does it slow down inference (making predictions on new instances)?

      What about MC Dropout?

    A.

      Dropout speeds up training as a large averaging ensemble of smaller networks is trained

      Model is evaluated with dropout disabled. Has no effect on trained network's predictions.

      MC Dropout extends dropout (calculating prediction/uncertainties) 

'''