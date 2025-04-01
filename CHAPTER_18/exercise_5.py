"""

    Q.
        What is the credit assignment problem. When does it occur. How can your alleviate it?

    A.
        When an agent gets a reward, it is hard to know which action should be credited for it. 

        The problem is tackled by evaluating the sum of all rewards that come after a reward with an applied discount factor at each step. 

        The sum of discounted rewards is called an action return. 

        Next, we estimate how much better or worse an action is, compared to the other possible actions on average(called action advantage). 

"""