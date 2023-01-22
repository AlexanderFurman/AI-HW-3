from copy import deepcopy
import random
import numpy as np


def value_iteration(mdp, U_init, epsilon=10 ** (-3)):
    # TODO:
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the U obtained at the end of the algorithms' run.
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================


def get_policy(mdp, U):
    # TODO:
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================


def q_learning(mdp, init_state, total_episodes=10000, max_steps=999, learning_rate=0.7, epsilon=1.0,
                      max_epsilon=1.0, min_epsilon=0.01, decay_rate=0.8):
    # TODO:
    # Given the mdp and the Qlearning parameters:
    # total_episodes - number of episodes to run for the learning algorithm
    # max_steps - for each episode, the limit for the number of steps
    # learning_rate - the "learning rate" (alpha) for updating the table values
    # epsilon - the starting value of epsilon for the exploration-exploitation choosing of an action
    # max_epsilon - max value of the epsilon parameter
    # min_epsilon - min value of the epsilon parameter
    # decay_rate - exponential decay rate for exploration prob
    # init_state - the initial state to start each episode from
    # return: the Qtable learned by the algorithm
    #

    # ====== YOUR CODE: ======

    #initialize qtable with all zeros
    q_table = np.zeros((mdp.num_row*mdp.num_col, len(mdp.actions)))

    #place values of terminal states in q table
    for terminal_state in mdp.terminal_states:
        for i in range(len(mdp.actions)):
            q_table[q_index(mdp, terminal_state),i] = mdp.board[terminal_state[0]][terminal_state[1]]

    for episode in range(total_episodes):
        #Reset to initial state
        state = init_state
        done = False

        for _ in range(max_steps):

            #choose action (a) in current world state (s)
            #choose random number from 0-1
            exp_exp_tradeoff = np.random.uniform(0,1)

            #if this number > epsilon --> exploitation (taking biggest Q value for this state)
            if exp_exp_tradeoff > epsilon:
                desired_action_index = np.argmax(q_table[q_index(mdp, state),:])
            
            #otherwise, pick random action
            else:
                desired_action_index = np.random.randint(0,4)

            desired_action = list(mdp.actions)[desired_action_index] # gives key --> 'UP', 'DOWN', etc.
            
            #given that there is uncertainty in the actions we find the true action taken, using the transition function
            action = np.random.choice(list(mdp.actions.keys()), p=mdp.transition_function[desired_action])
            action_index = list(mdp.actions).index(action)

            #take action (a), and find the new state (s'), and reward (r)
            new_state, reward, done = take_next_step(mdp, state, action)

            # update Q(s,a) := Q(s,a) + lr[R(s,a) + gamma * max Q(s',a') - Q(s,a)] 
            q_table[q_index(mdp, state), action_index] += learning_rate * (reward + decay_rate * np.max(q_table[q_index(mdp, new_state),:]) - q_table[q_index(mdp, state), action_index])

            #update state to be new state
            state = new_state

            #if done, finish episode:
            if done == True:
                break
        
        #decrease epsilon --> ensures we explore less over time
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
    
    return q_table


    # ========================


def q_table_policy_extraction(mdp, qtable):
    # TODO:
    # Given the mdp and the Qtable:
    # return: the policy corresponding to the Qtable
    #

    # ====== YOUR CODE: ======
    # raise NotImplementedError
    policy_table = [[0]*mdp.num_col for _ in range(mdp.num_row)]
    for q_index in range(mdp.num_row*mdp.num_col):
        desired_action_index = np.argmax(qtable[q_index,:])
        desired_action = list(mdp.actions)[desired_action_index]
        i, j = state_from_index(mdp, q_index)
        policy_table[i][j] = desired_action
    
    return policy_table
    # ========================


# BONUS

def policy_evaluation(mdp, policy):
    # TODO:
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #

    # ====== YOUR CODE: ======
    # raise NotImplementedError
    raise NotImplementedError
    # ========================


def policy_iteration(mdp, policy_init):
    # TODO:
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================


# ====== HELPER FUNCTIONS: ======

def q_index(mdp, state_tuple):
    return state_tuple[0]*mdp.num_col + state_tuple[1]

def state_from_index(mdp, q_index):
    return int(q_index/mdp.num_col), int(q_index%mdp.num_col)

def take_next_step(mdp, state, action):
    new_state = mdp.step(state, action)
    reward = float(mdp.board[new_state[0]][new_state[1]])
    done = new_state in mdp.terminal_states

    return new_state, reward, done

# ===============================