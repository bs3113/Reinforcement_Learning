import numpy as np
from RLalgs.utils import epsilon_greedy
import random

def SARSA(env, num_episodes, gamma, lr, e):
    """
    Implement the SARSA algorithm following epsilon-greedy exploration. Update Q at the end of every episode.

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
            State-action values
    """
    
    Q = np.zeros((env.nS, env.nA))
    
    #TIPS: Call function epsilon_greedy without setting the seed
    #      Choose the first state of each episode randomly for exploration.
    ############################
    # YOUR CODE STARTS HERE
    
    for i in range(num_episodes):
        s = action = np.random.random_integers(env.nS - 1)
        a = epsilon_greedy(Q[s], e, seed = None)
        t = False
        while not t:
            next_s, r, t = env.P[s][a][0][1], env.P[s][a][0][2], env.P[s][a][0][3]
            next_a = epsilon_greedy(Q[next_s], e, seed = None)
            Q[s][a] = Q[s][a] + lr * (r + gamma * Q[next_s][next_a] - Q[s][a])
            s = next_s
            a = next_a
    # YOUR CODE ENDS HERE
    ############################

    return Q