# cart_pole 状態を行動に反映させてみるテスト
# 角度が負(左に傾いている状態)なら左に行動
# 角度が正(右に傾いている状態)なら右に行動する
# 振動が大きくなっていき、安定しない。

import gym
import time
import numpy as np
M = 20

def actionByAngle(observation):
	"""角度で行動決定する単純なやつ"""
	angle = observation[2]
	action = 1 if angle > 0 else 0
	return action

def getAction(observation,w_s):
	"""状態から行動決定"""
	return actionByAngle(observation)


if __name__ == '__main__':

	env = gym.make('CartPole-v0')
	total_reward_sum = np.zeros(M)
	for i_episode in range(M):
		observation = env.reset()
		total_reward = 0
		for t in range(100):
			env.render()
			# time.sleep(0.3)

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
