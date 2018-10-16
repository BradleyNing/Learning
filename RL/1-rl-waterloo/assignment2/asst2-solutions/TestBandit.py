import numpy as np
import MDP
import RL2
import matplotlib.pyplot as plt


def sampleBernoulli(mean):
    ''' function to obtain a sample from a Bernoulli distribution

    Input:
    mean -- mean of the Bernoulli
    
    Output:
    sample -- sample (0 or 1)
    '''

    if np.random.rand(1) < mean: return 1
    else: return 0



# Multi-arm bandit problems (3 arms with probabilities 0.3, 0.5 and 0.7)
T = np.array([[[1]],[[1]],[[1]]])
R = np.array([[0.3],[0.5],[0.7]])
discount = 0.999
mdp = MDP.MDP(T,R,discount)
banditProblem = RL2.RL2(mdp,sampleBernoulli)


nTrials = 1000

# Test epsilon greedy strategy
eGreedy_rewards = np.zeros((nTrials, 200))
for trial in range(nTrials):
    [empiricalMeans, eGreedy_rewards[trial,:]]  = banditProblem.epsilonGreedyBandit(nIterations=200)
print "\nepsilonGreedyBandit results"
print empiricalMeans

# Test Thompson sampling strategy
TS_rewards = np.zeros((nTrials, 200))
for trial in range(nTrials):
    [empiricalMeans, TS_rewards[trial,:]] = banditProblem.thompsonSamplingBandit(prior=np.ones([mdp.nActions,2]),nIterations=200)
print "\nthompsonSamplingBandit results"
print empiricalMeans

# Test UCB strategy
UCB_rewards = np.zeros((nTrials, 200))
for trial in range(nTrials):
    [empiricalMeans, UCB_rewards[trial,:]] = banditProblem.UCBbandit(nIterations=200)
print "\nUCBbandit results"
print empiricalMeans

plt.figure(figsize=(12,9))
plt.plot(range(200),np.mean(eGreedy_rewards, axis=0), label='eGreedy')
plt.plot(range(200),np.mean(TS_rewards, axis=0), label='Thomson Sampling')
plt.plot(range(200),np.mean(UCB_rewards, axis=0), label='UCB')



plt.grid()
plt.legend()
plt.title("Average Reward vs Episode")
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.show()
