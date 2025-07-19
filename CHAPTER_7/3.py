'''
    Q.

      Is it possible to speed up training of a bagging ensemble by distributing it across multiple servers?

      What about pasting ensembles, boosting ensembles, Random Forests, or stacking ensembles?


    A.

      Yes, running a bagging ensemble on parallel cpus or distributed servers will speed up training 

      Yes, Random Forest improves with distributed servers 

      Yes, stacking ensembles can be distributed across miltiple servers 

      Pasting ensemble will be complicated because samples are not replaced. Prior to using multiple servers to speed up computation,  the training data must be split accordingly for each server

      Boosting ensembles cannot be distributed across mutiple servers. Why? Each process in pipeline is dependant on the previous process. 

      

'''