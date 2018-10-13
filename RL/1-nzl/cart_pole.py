# -*- coding: utf-8 -*-
# test for CartPole
# adapt from the book hands on ML with SL and TF
# by nzl 2018/09/18

import gym
import time

env = gym.make('CartPole-v0')
obs = env.reset()
print (obs)
env.render()

def basic_policy(obs):
	angle = obs[2]
	return 0 if angle < 0 else 1

totals = []
for episode in range(10):
	episode_rewards = 0
	obs = env.reset()
	for step in range(100): # 1000 steps max, we don't want to run forever
		action = basic_policy(obs)
		obs, reward, done, info = env.step(action)
		episode_rewards += reward
		env.render()
		if done:
			time.sleep(0.5)
			break
	totals.append(episode_rewards)

env.close()
import numpy as np
print(np.mean(totals), np.std(totals), np.min(totals), np.max(totals))