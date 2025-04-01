"""
    Q.

        What is an off-policy RL algorithm?

    A.

        A policy being trained that is not being executed. For example, a Data Scientist may created a custom algorithm to control actions taken by the agent, while some new metric is being generated/trained to understand the optimal policy taken by the agent. 
        New Metric - Not executed but experiences are used to train metric. 
        Custom Algorithm - Agent called to play actions 

        Conversly, an example would be policy gradient algorithm. It explores the world using the policy (DNN) being trained.

"""