"""
This is a vanilla model for GRN controlling which is based on
Q-learning, that is to say it can only handle small space state
problem.
Here I would take n=3 for example.
the
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

np.random.seed(2)  # reproducible

N_GENES = 3
N_STATES = 2 ** N_GENES   # the length of the 1 dimensional world
ACTIONS = ['still', 'inverse']     # available actions
EPSILON = 0.9   # greedy police
ALPHA = 0.1     # learning rate
GAMMA = 0.9    # discount factor
MAX_EPISODES = 13   # maximum episodes
FRESH_TIME = 0.03    # fresh time for one move
ATTRACTORS = '000'  # attractors

def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),     # q_table initial values
        columns=actions,    # actions's name
    )
    # print(table)    # show table
    return table


def choose_action(state, q_table):
    # This is how to choose an action
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):  # act non-greedy or state-action have no value
        action_name = np.random.choice(ACTIONS)
    else:   # act greedy
        # action_name = state_actions.argmax()
        action_name = state_actions.idxmax()
        #  replace argmax to idxmax as argmax means
        #  a different function in newer version of pandas
    return action_name


def get_env_feedback(S, A):
    # This is how agent will interact with the environment
    S_bin = "{:03b}".format(S)
    a_idx = np.random.randint(0, N_GENES)
    if A == 'inverse':    # inverse
        if S_bin == '001' and a_idx == 2:   # terminate
            S_ = ATTRACTORS
            R = 1
        elif S_bin == '010' and a_idx == 1:
            S_ = ATTRACTORS
            R = 1
        elif S_bin == '100' and a_idx == 0:
            S_ = ATTRACTORS
            R = 1
        else:
            if S_bin[a_idx] == 0:
                S_bin = S_bin[:a_idx] + '1' + S_bin[a_idx+1:]
                S_ = int(S_bin, 2)
                R = 0
            else:
                S_bin = S_bin[:a_idx] + '0' + S_bin[a_idx+1:]
                S_ = int(S_bin, 2)
                R = 0
    else:   # not inverse
        R = 0
        S_ = S
    return S_, R


def update_env(S, episode, step_counter):
    # This is how environment be updated
    # env_list = ['_']*(N_STATES-1) + ['T']   # '---------T' our environment
    env_list = ['_']*(N_STATES)
    if S == ATTRACTORS:
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def rl():
    # main part of RL loop
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = np.random.randint(1, N_STATES)
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:

            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S, A)  # take action & get next state and reward
            q_predict = q_table.loc[S, A]
            if S_ != ATTRACTORS:
                q_target = R + GAMMA * q_table.iloc[S_, :].max()  # next state is not terminal
            else:
                q_target = R     # next state is terminal
                is_terminated = True    # terminate this episode

            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  # update
            S = S_  # move to next state

            update_env(S, episode, step_counter+1)
            step_counter += 1

    return q_table


if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)
    # print(R_list)
    #plt.plot(range(MAX_EPISODES), R_list)
    #plt.show()

