# cart_pole 
# 離散的に分解してMDPに適用、Q学習をしてみる。

import gym
import time
import numpy as np

M = 20

def getAction(observation,w_s):
	"""状態から行動決定"""
	# 角度で右か左か決める単純なやつ
	v =  np.dot(observation, w_s)
	if v > 0:
		action = 1
	else:
		action = 0
	print(v,action)
	return action


if __name__ == '__main__':

	# 行動価値関数。こいつを学習させる。
	# stateを離散的に定義するべき。
	# Q(s,a)は{(s,a):value,(s,a):value,..}というような形式。
	Q = {}

	env = gym.make('CartPole-v0')
	total_reward_sum = np.zeros(M)
	for i_episode in range(M):
		observation = env.reset()
		total_reward = 0
		for t in range(100):
			env.render()

			action = getAction(observation,w_s)

			observation, reward, done, info = env.step(action)
			total_reward += 1

			if done:
				print("Episode finished after {} timesteps".format(t+1))
				print("total_reward is {0}".format(total_reward))
				total_reward_sum[i_episode] = total_reward
				if i_episode == M-1:
					print("all total reward is {0}".format(total_reward_sum))
				break
