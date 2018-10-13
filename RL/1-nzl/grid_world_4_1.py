# -*- coding: utf-8 -*-
# DP value iteration
# adapt from the book RL introduction ch4, fig4.1
# by nzl 2018/09/19
import numpy as np

WORLD_SIZE = 4
T1_POS = [0, 0]
T2_POS = [3, 3]

# left, up, right, down
ACTIONS = [np.array([0, -1]),
           np.array([-1, 0]),
           np.array([0, 1]),
           np.array([1, 0])]
ACTION_PROB = 0.25

def step(state, action):
    reward = -1
    finished = False

    if state == T1_POS or state == T2_POS:
        reward = 0
        finished = True
        return state, reward #, finished
    
    state = np.array(state)
    next_state = (state + action).tolist()
    x, y = next_state
    if x < 0 or x >= WORLD_SIZE or y < 0 or y >= WORLD_SIZE:
        next_state = state

    return next_state, reward #, finished

def get_v_value():
    value = np.zeros((WORLD_SIZE, WORLD_SIZE))
    iteration = 0
    while True:
        # keep iteration until convergence
        new_value = np.zeros(value.shape)
        for i in range(0, WORLD_SIZE):
            for j in range(0, WORLD_SIZE):
                for action in ACTIONS:
                    (next_i, next_j), reward = step([i, j], action)
                    # bellman equation
                    new_value[i, j] += ACTION_PROB * (reward + value[next_i, next_j])
        
        # show the first 4 iteration
        if iteration < 4 or iteration == 10:
            t_value = np.round(value, decimals=1)
            print (t_value)
        iteration += 1

        if np.sum(np.abs(value - new_value)) < 1e-4:
            new_value = np.round(new_value, decimals=2)
            return new_value

        value = new_value.copy()


def get_v_star():
    value = np.zeros((WORLD_SIZE, WORLD_SIZE))
    while True:
        # keep iteration until convergence
        new_value = np.zeros(value.shape)
        for i in range(0, WORLD_SIZE):
            for j in range(0, WORLD_SIZE):
                values = []
                for action in ACTIONS:
                    (next_i, next_j), reward = step([i, j], action)
                    # # value iteration
                    values.append(reward+DISCOUNT*value[next_i, next_j])
                new_value[i, j] = np.max(values)
        
        if np.sum(np.abs(value - new_value)) < 1e-4:
            new_value = np.round(new_value, decimals=2)
            return new_value

        value = new_value

v_values = get_v_value()
print (v_values)

# v_star = get_v_star()
# print(v_star)
