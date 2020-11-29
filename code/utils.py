import numpy as np

def initialize_values(state_space):
    """
    Value array for states (1D)
    """
    values = np.zeros(len(state_space))
    return values

def initialize_rewards(state_space):
    """
    Create a R(s,'s) matrix (reward can be thought of as independent of action)
    The reward is equal to the increase in the number of agents influenced
    """
    R = np.zeros((len(state_space), len(state_space)))
    for i, state1 in enumerate(state_space):
        for j, state2 in enumerate(state_space):
            reward = np.max((0, (np.sum(state2) - np.sum(state1))))
            R[i,j] = reward
    return R
    
def calculate_policy_value(env, policy, values, rewards, gamma = 0.85, epsilon=0.01):
    """

    """
    new_values = values.copy()
    delta = 1e9 # arbitrarily large number
    counter = 0

    while True:
        deltas = []
        for state_index, state in enumerate(env.state_space):
            
            # Extract the values relevant to the current state
            cur_value = new_values[state_index]
            cur_action_index = policy[state_index]
            
            transition_matrix = env.T[cur_action_index,state_index,:]
            reward_matrix = rewards[state_index,:].reshape(-1,)

            # Calculate the next value using Bellman update
            next_value = np.matmul(transition_matrix, (reward_matrix + gamma * new_values))
            
            # Update the value matrix 
            new_values[state_index] = next_value
            deltas.append(abs(next_value - cur_value))

        counter += 1
        if counter % 20 == 0:
            print(f"{counter} iterations run - max delta = {np.max(deltas)}")
        if np.max(deltas) < epsilon:
            break

    return new_values

def calculate_policy_improvement(env, policy, values, rewards, gamma=0.85):
    """

    """
    stable_policy=True
    new_policy = policy.copy()

    for state_index, state in enumerate(env.state_space):

        old_policy = policy.copy()

        # Calculate the value of taking a specific action followed by the
        # original policy
        action_values = []
        for action_index, action in enumerate(env.action_space):
            action_value = np.matmul(env.T[action_index, state_index, :], 
                                     (rewards[state_index,:] + gamma * values))
            action_values.append(action_value)
        
        best_action = np.argmax(action_values)

        # Update the policy
        new_policy[state_index] = best_action

        if new_policy[state_index] != old_policy[state_index]:
            stable_policy = False

    return new_policy, stable_policy

def policy_iteration(env, policy, values, rewards, gamma = 0.85, epsilon=0.01):
    """

    """
    stable = False
    new_policy = policy.copy()
    new_values = values.copy()

    while stable == False:
        new_values = calculate_policy_value(env, new_policy, new_values, rewards, gamma = gamma, epsilon=epsilon)
        new_policy, stable = calculate_policy_improvement(env, new_policy, new_values, rewards, gamma= gamma) 

    return new_policy       