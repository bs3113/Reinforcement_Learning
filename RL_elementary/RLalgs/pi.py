import numpy as np
from RLalgs.utils import action_evaluation

def policy_iteration(env, gamma, max_iteration, theta):
    """
    Implement Policy iteration algorithm.

    Inputs:
    env: OpenAI Gym environment.
            env.P: dictionary
                    P[state][action] is list of tuples. Each tuple contains probability, nextstate, reward, terminal
                    probability: float
                    nextstate: int
                    reward: float
                    terminal: boolean
            env.nS: int
                    number of states
            env.nA: int
                    number of actions
    gamma: float
            Discount factor.
    max_iteration: int
            The maximum number of iterations to run before stopping.
    theta: float
            The threshold of convergence.
            
    Outputs:
    V: numpy.ndarray
    policy: numpy.ndarray
    numIterations: int
    """

    V = np.zeros(env.nS)
    policy = np.zeros(env.nS, dtype = np.int32)
    policy_stable = False
    numIterations = 0
    
    while not policy_stable and numIterations < max_iteration:
        #Implement it with function policy_evaluation and policy_improvement
        ############################
        # YOUR CODE STARTS HERE
        policy, policy_stable = policy_improvement(env, V, policy, gamma)
        if not policy_stable:
            V = policy_evaluation(env, policy, gamma, theta)
        # YOUR CODE ENDS HERE
        ############################
        numIterations += 1
        
    return V, policy, numIterations


def policy_evaluation(env, policy, gamma, theta):
    """
    Evaluate the value function from a given policy.

    Inputs:
    env: OpenAI Gym environment.
            env.P: dictionary
                    
            env.nS: int
                    number of states
            env.nA: int
                    number of actions

    gamma: float
            Discount factor.
    policy: numpy.ndarray
            The policy to evaluate. Maps states to actions.
    theta: float
            The threshold of convergence.
    
    Outputs:
    V: numpy.ndarray
            The value function from the given policy.
    """
    ############################
    # YOUR CODE STARTS HERE
    V = np.zeros(env.nS)
    
    while 1:
        delta = 0
        for s in range(env.nS):
            temp = V[s]
            a = policy[s]
            this_v = 0
            for t in env.P[s][a]:
                p, next_s, r, terminate = t[0],t[1],t[2],t[3]
                this_v += p*(r + gamma * V[next_s])
            V[s] = this_v
            delta = max(delta, abs(temp - V[s]))
        if delta < theta:
            break
    # YOUR CODE ENDS HERE
    ############################

    return V


def policy_improvement(env, value_from_policy, policy, gamma):
    """
    Given the value function from policy, improve the policy.

    Inputs:
    env: OpenAI Gym environment
            env.P: dictionary
                    P[state][action] is tuples with (probability, nextstate, reward, terminal)
                    probability: float
                    nextstate: int
                    reward: float
                    terminal: boolean
            env.nS: int
                    number of states
            env.nA: int
                    number of actions

    value_from_policy: numpy.ndarray
            The value calculated from the policy
    policy: numpy.ndarray
            The previous policy.
    gamma: float
            Discount factor.

    Outputs:
    new policy: numpy.ndarray
            An array of integers. Each integer is the optimal action to take
            in that state according to the environment dynamics and the
            given value function.
    policy_stable: boolean
            True if the "optimal" policy is found, otherwise false
    """
    ############################
    # YOUR CODE STARTS HERE
    temp = policy.copy()
    policy_stable = True
    for s in range(env.nS):
        policies = []
        for a in range(env.nA):
            this_p = 0
            for t in env.P[s][a]:
                p, next_s, r, terminate = t[0],t[1],t[2],t[3]
                this_p += p * (r + gamma * value_from_policy[next_s])
            policies.append(this_p)
        policy[s] = np.argmax(policies)
    if np.any(policy != temp):
            policy_stable = False
    # YOUR CODE ENDS HERE
    ############################

    return policy, policy_stable