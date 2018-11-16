import numpy as np
from RLalgs.utils import epsilon_greedy
import random
def QLearning(env, num_episodes, gamma, lr, e):
    """
    Implement the Q-learning algorithm following the epsilon-greedy exploration. Update Q at the end of every episode.

    Inputs:
    env: OpenAI Gym environment 
            env.P: dictionary
                    P[state][action] are tuples of tuples tuples with (probability, nextstate, reward, terminal)
                    probability: float
                    nextstate: int
                    reward: float
                    terminal: boolean
            env.nS: int
                    number of states
            env.nA: int
                    number of actions
    num_episodes: int
            Number of episodes of training
    gamma: float
            Discount factor.
    lr: float
            Learning rate.
    e: float
            Epsilon value used in the epsilon-greedy method.

    Outputs:
    Q: numpy.ndarray
    """

    Q = np.zeros((env.nS, env.nA))
    #TIPS: Call function epsilon_greedy without setting the seed
    #      Choose the first state of each episode randomly for exploration.
    ############################
    # YOUR CODE STARTS HERE
    for i in range(num_episodes):
        t = False
        s = action = np.random.random_integers(env.nS - 1)
        while not t:
            e_greedy_action = epsilon_greedy(Q[s], e, seed = None)
            a = e_greedy_action 

            next_s, r, t = env.P[s][a][0][1], env.P[s][a][0][2], env.P[s][a][0][3]
            maxq = max(Q[next_s])

            Q[s][a] = Q[s][a] + lr * (r + gamma * maxq - Q[s][a])
            s = next_s
        
    
    # YOUR CODE ENDS HERE
    ############################

    return Q