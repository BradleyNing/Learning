import numpy as np
from MDP_alg1 import MDP

''' Construct simple MDP as described in Lecture 2a Slides 13-14'''
# Transition function: |A| x |S| x |S'| array
T = np.array([[[0.5,0.5,0,0],[0,1,0,0],[0.5,0.5,0,0],[0,1,0,0]], \
			  [[1,0,0,0],[0.5,0,0,0.5],[0.5,0,0.5,0],[0,0,0.5,0.5]]])
			  
# Reward function: |A| x |S| array
R = np.array([[0,0,10,10], \
			  [0,0,10,10]])
			  
# Discount factor: scalar in [0,1)
discount = 0.9        

# MDP object
mdp = MDP(T,R,discount)

'''Test each procedure'''
[V,nIterations,epsilon] = mdp.valueIteration(initialV=np.zeros(mdp.nStates))
print('valueIteration(initialV=np.zeros(mdp.nStates)): (V, nIterations, epsilon)')
print(V, nIterations, epsilon)
print('\n')

policy = mdp.extractPolicy(V)
print('extractPolicy(V): (policy)')
print(policy)
print('\n')

V = mdp.evaluatePolicy(np.array([1, 0, 1, 0]))
print('evaluatePolicy(np.array([1, 0, 1, 0])): (V)')
print (V)
print('\n')

[policy,V,iterId] = mdp.policyIteration(np.array([0,0,0,0]))
print('policyIteration(np.array([0,0,0,0])): (policy,V,iterId)')
print(policy, V, iterId)
print('\n')

[V,iterId,epsilon] = mdp.evaluatePolicyPartially(np.array([1,0,1,0]), np.array([0,10,0,13]))
print('evaluatePolicyPartially(np.array([1,0,1,0]), np.array([0,10,0,13])): (V, iterId, epsilon)')
print(V, iterId, epsilon)
print('\n')

[policy,V,iterId,tolerance] = mdp.modifiedPolicyIteration(np.array([1,0,1,0]), np.array([0,10,0,13]))
#[policy,V,iterId,tolerance] = mdp.modifiedPolicyIteration(np.array([1,0,1,0]), np.array([0,0,0,1]))
print('modifiedPolicyIteration(np.array([1,0,1,0]), np.array([0,10,0,13])): (policy,V,iterId,tolerance)')
print(policy,V,iterId,tolerance)
print('-----------------end of the program--------------------')
