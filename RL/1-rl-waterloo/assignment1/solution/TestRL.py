import numpy as np
import MDP
import RL


''' Construct simple MDP as described in Lecture 2a Slides 13-14'''
T = np.array([[[0.5,0.5,0,0],[0,1,0,0],[0.5,0.5,0,0],[0,1,0,0]],[[1,0,0,0],[0.5,0,0,0.5],[0.5,0,0.5,0],[0,0,0.5,0.5]]])
R = np.array([[0,0,10,10],[0,0,10,10]])
discount = 0.9        
mdp = MDP.MDP(T,R,discount)
rlProblem = RL.RL(mdp,np.random.normal)

# Test Q-learning 
[Q,policy] = rlProblem.qLearning(s0=0, initialQ=np.zeros([mdp.nActions,mdp.nStates]),nEpisodes=500,nSteps=50,epsilon=0.0)
                       #qLearning(self,s0,initialQ,nEpisodes,nSteps,epsilon=0,temperature=0)
print ("\nQ-learning results")
print (Q)
print (policy)

#print np.amax(Q, axis=0)
