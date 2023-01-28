from copy import deepcopy
import random
import numpy as np



def next_state_utility_calc(mdp,U,state,direction):
    next_state_utility=0
    if direction == 0:
        next_state_utility = U[mdp.step(state, "UP")[0]][mdp.step(state, "UP")[1]]
    if direction == 1:
        next_state_utility = U[mdp.step(state, "DOWN")[0]][mdp.step(state, "DOWN")[1]]
    if direction == 2:
        next_state_utility = U[mdp.step(state, "RIGHT")[0]][mdp.step(state, "RIGHT")[1]]
    if direction == 3:
        next_state_utility = U[mdp.step(state, "LEFT")[0]][mdp.step(state, "LEFT")[1]]
    return next_state_utility

def dir_index_to_str(direction):
    if(direction == 0):
        return "UP"
    if (direction == 1):
        return "DOWN"
    if (direction == 2):
        return "RIGHT"
    if (direction == 3):
        return "LEFT"

def value_iteration(mdp, U_init, epsilon=10 ** (-3)):
    # TODO:
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the U obtained at the end of the algorithms' run.
    #

    # ====== YOUR CODE: ======
    U = np.array(U_init,dtype=float,copy=True)
    U_tag = np.array(U_init,dtype=float,copy=True)
    states = [(row,col) for row in range(0,mdp.num_row) for col in range(0,mdp.num_col) if (mdp.board[row][col] != 'WALL')]
    while (1):
        U = np.array(U_tag,dtype=float,copy=True)
        delta = 0
        for state in states:
            max_value=0
            for action in mdp.actions:
                if state not in mdp.terminal_states:
                    curr_sum=0
                    for direction in range(0,4):
                        curr_sum += next_state_utility_calc(mdp,U,state,direction)*mdp.transition_function[action][direction]
                    max_value = max(max_value,curr_sum)
            U_tag[state[0]][state[1]] = float(mdp.board[state[0]][state[1]])
            if (state[0],state[1]) not in mdp.terminal_states:
                U_tag[state[0]][state[1]] += mdp.gamma*max_value
            delta = max(abs(U_tag[state[0]][state[1]]-U[state[0]][state[1]]),delta)
        if delta < epsilon*(1-mdp.gamma)/mdp.gamma:
            break
    return U
    # ========================


def get_policy(mdp, U):
    # TODO:
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #

    # ====== YOUR CODE: ======
    policy_table = np.empty([mdp.num_row, mdp.num_col], dtype=str)
    states = [(row,col) for row in range(0,mdp.num_row) for col in range(0,mdp.num_col) if (mdp.board[row][col] != 'WALL' and (row,col) not in mdp.terminal_states)]
    for state in states:
        up_util = next_state_utility_calc(mdp,U,state,0)
        down_util = next_state_utility_calc(mdp,U,state,1)
        right_util = next_state_utility_calc(mdp,U,state,2)
        left_util = next_state_utility_calc(mdp,U,state,3)
        direction = np.argmax(np.array([up_util*mdp.transition_function["UP"][0] + down_util*mdp.transition_function["UP"][1] + right_util*mdp.transition_function["UP"][2] + left_util*mdp.transition_function["UP"][3],
                                        up_util*mdp.transition_function["DOWN"][0] + down_util*mdp.transition_function["DOWN"][1] + right_util*mdp.transition_function["DOWN"][2] + left_util*mdp.transition_function["DOWN"][3],
                                        up_util*mdp.transition_function["RIGHT"][0] + down_util*mdp.transition_function["RIGHT"][1] + right_util*mdp.transition_function["RIGHT"][2] + left_util*mdp.transition_function["RIGHT"][3],
                                        up_util*mdp.transition_function["LEFT"][0] + down_util*mdp.transition_function["LEFT"][1] + right_util*mdp.transition_function["LEFT"][2] + left_util*mdp.transition_function["LEFT"][3]]))
        if (state in mdp.terminal_states):
            policy_table[state[0]][state[1]] = mdp.board[state[0]][state[1]]
        else:
            policy_table[state[0]][state[1]] = dir_index_to_str(direction)
    return policy_table
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

    # place values of terminal states in q table
    for terminal_state in mdp.terminal_states:
        for i in range(len(mdp.actions)):
            q_table[q_index(mdp, terminal_state),i] = mdp.board[terminal_state[0]][terminal_state[1]]
            q_table[q_index(mdp, terminal_state),i] = mdp.board[terminal_state[0]][terminal_state[1]]

    for episode in range(total_episodes):
        #Reset to initial state
        state = init_state
        done = False

        for _ in range(max_steps):

            #choose action (a) in current world state (s)
            
            #pick random number between min_epsilon and max_epsilon
            exp_exp_tradeoff = np.random.uniform(min_epsilon, max_epsilon, 1)

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