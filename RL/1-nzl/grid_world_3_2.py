# -*- coding: utf-8 -*-
# test for CartPole
# adapt from the book RL introduction ch3, fig3.2
# by nzl 2018/09/18
import numpy as np

WORLD_SIZE = 5
A_POS = [0, 1]
A_PRIME_POS = [4, 1]
B_POS = [0, 3]
B_PRIME_POS = [2, 3]
DISCOUNT = 0.9
CONST = 0
T_CONST = 0

# left, up, right, down
ACTIONS = [np.array([0, -1]),
           np.array([-1, 0]),
           np.array([0, 1]),
           np.array([1, 0])]
ACTION_PROB = 0.25

def step(state, action):
    if state == A_POS:
        return A_PRIME_POS, 10+CONST+T_CONST
    if state == B_POS:
        return B_PRIME_POS, 5+CONST+T_CONST

    state = np.array(state)
    next_state = (state + action).tolist()
    x, y = next_state
    if x < 0 or x >= WORLD_SIZE or y < 0 or y >= WORLD_SIZE:
        reward = -1.0+CONST
        next_state = state
    else:
        reward = 0+CONST
    return next_state, reward

def get_v_value():
    value = np.zeros((WORLD_SIZE, WORLD_SIZE))
    while True:
        # keep iteration until convergence
        new_value = np.zeros(value.shape)
        for i in range(0, WORLD_SIZE):
            for j in range(0, WORLD_SIZE):
                for action in ACTIONS:
                    (next_i, next_j), reward = step([i, j], action)
                    # bellman equation
                    new_value[i, j] += ACTION_PROB * (reward + DISCOUNT * value[next_i, next_j])
        
        if np.sum(np.abs(value - new_value)) < 1e-4:
            new_value = np.round(new_value, decimals=2)
            return new_value

        value = new_value

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

v_star = get_v_star()
print(v_star)
