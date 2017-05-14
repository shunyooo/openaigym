# 価値関数近似をやってみる
# 線形モデルを用いた線形近似モデルの例。
# 最良のパラメータを推定するのに、ランダムで探索する方法。
# cart_pole 

import gym
import time
import numpy as np
import random

env = gym.make('CartPole-v0')

ACTIONS = [0,1]
n_action = len(ACTIONS)
n_states = 4

def run_episode(env, parameters, render = False):  
	"""
	全てのステップに対し、ポールをまっすぐに保ち、+1報酬を得る。
	与えられた重みのセットがどれほど良いかを推定するには、
	ポールが倒れるまでエピソードを走らせ、
	どれだけ報酬が得られたかを評価する必要がある。
	"""
	observation = env.reset()
	totalreward = 0
	for _ in range(200):
		if render:
			env.render()
		action = 0 if np.matmul(parameters,observation) < 0 else 1
		observation, reward, done, info = env.step(action)
		totalreward += reward
		if done:
			break
	return totalreward# 報酬和を返す。


if __name__ == '__main__':

	bestparams = None  
	bestreward = 0  
	for _ in range(10000):  

		# -1 から 1の間でランダムに初期化
		# 状態の次元数だけ用意
		parameters = np.random.rand(4) * 2 - 1

		reward = run_episode(env,parameters)
		if reward > bestreward:
			bestreward = reward
			bestparams = parameters
			# considered solved if the agent lasts 200 timesteps
			if reward == 200:
				break

	for _ in range(3):
		run_episode(env,bestparams,render = True)



