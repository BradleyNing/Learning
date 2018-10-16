import numpy as np
from MDP_alg1 import MDP
import random

class RL:
    def __init__(self, mdp, sampleReward):
        '''Constructor for the RL class

        Inputs:
        mdp -- Markov decision process (T, R, discount)
        sampleReward -- Function to sample rewards (e.g., bernoulli, Gaussian).
        This function takes one argument: the mean of the distributon and 
        returns a sample from the distribution.
        '''

        self.mdp = mdp
        self.sampleReward = sampleReward

    def sampleRewardAndNextState(self, state, action):
        '''Procedure to sample a reward and the next state
        reward ~ Pr(r)
        nextState ~ Pr(s'|s,a)

        Inputs:
        state -- current state
        action -- action to be executed

        Outputs: 
        reward -- sampled reward
        nextState -- sampled next state
        '''

        reward = self.sampleReward(self.mdp.R[action, state])
        cumProb = np.cumsum(self.mdp.T[action,state,:])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward, nextState]

    def qLearning(self,s0,initialQ,nEpisodes,nSteps,epsilon=0,temperature=0):
        '''qLearning algorithm.  Epsilon exploration and Boltzmann exploration
        are combined in one procedure by sampling a random action with 
        probabilty epsilon and performing Boltzmann exploration otherwise.  
        When epsilon and temperature are set to 0, there is no exploration.

        Inputs:
        s0 -- initial state
        initialQ -- initial Q function (|A|x|S| array)
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- # of steps per episode
        epsilon -- probability with which an action is chosen at random
        temperature -- parameter that regulates Boltzmann exploration

        Outputs: 
        Q -- final Q function (|A|x|S| array)
        policy -- final policy
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        print('epsilon: ', epsilon)
        Q = initialQ.copy()
        policy = np.zeros(self.mdp.nStates, int)
        counter = np.ones([self.mdp.nActions, self.mdp.nStates])
        np.random.seed(seed=1)
        deltas = np.zeros(nEpisodes)
        rewards = np.zeros(nEpisodes)
        for i in range(nEpisodes):
            s = s0     
            Q_old = Q.copy()       
            for j in range(nSteps):
                #if np.random.rand(1) <= epsilon:
                if np.random.uniform() < epsilon:
                    a = np.random.randint(self.mdp.nActions)
                else:
                    a = np.argmax(Q[:, s])

                counter[a,s] += 1
                [r, next_s] = self.sampleRewardAndNextState(s, a)
                rewards[i] += r*(self.mdp.discount**j)
                max_q_sa_p = max(Q[:, next_s])
                Q[a,s] += 1.0/counter[a,s] * (r+self.mdp.discount*max_q_sa_p-Q[a][s])
                s = next_s
        
            deltas[i] = np.abs(Q-Q_old).sum()/(self.mdp.nActions*self.mdp.nStates)

        #print('delta: ', delta)

        for s in range(self.mdp.nStates):
            policy[s] = np.argmax(Q[:, s])

        return [Q, policy, deltas, rewards]    