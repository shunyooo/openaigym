# cart_pole 
# 離散的に分解してMDPに適用、Q学習をしてみる。

import gym
import time
import numpy as np
import random


# 各状態を離散的に分割する定義
DISCRETE = {'pos':[-2.4,2.4,10],
			'cart_vec':[-3,3,10],
			'angle':[-2,2,10],
			'pole_vec':[-3.5,3.5,10]}

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

def action_e_greedy(observation,greedy_ratio,actions,Q):
	"""ε-greedy法で行動を決定"""
	if greedy_ratio > random.random():
		#ランダムに行動選択
		return random.choice(actions)
	else:
		#greedyに行動選択
		return action_greedy(observation,actions,Q)

def action_greedy(observation,actions,Q):
	"""Q(s,a)を比較してgreedy法で行動を決定。"""
	best_actions = []  #最高の行動が複数存在した場合
	max_q_value = -100000 #最大の行動価値を保存
	for a in actions:#すごく単純な最大求めるやつ
		q_value = get_Qvalue(observation,a,Q)
		if q_value > max_q_value:
			best_actions = [a,]
			max_q_value = q_value
		elif q_value == max_q_value:
			best_actions.append(a)
	return random.choice(best_actions)#Q値の最大値が複数存在する場合はその中からランダムに選択。

def get_Qvalue(observation,action,Q):
	"""Q(s,a)を取得。ここでのstateは連続値なので、揃える必要あり。"""
	state = state_from_observation(observation)
	return Q[state,action]

def state_from_observation(observation):
	"""連続値のobservationから離散値に区切ったstateを得る。"""
	pos,cart_vec,angle,pole_vec = observation
	# print(DISCRETE.values())
	state = tuple(discrete(v[0],v[1],v[2],prop) for (prop,v) in zip(observation,DISCRETE.values()))
	return state

def action_and_update_Qvalue(observation,action,Q,env,actions,t):
	alpha = 0.2
	gamma = 0.95

	state = state_from_observation(observation)
	"""sにおいて選択した行動aから、Q(s,a)を更新"""
	# 更新式:
	#       Q(s, a) <- Q(s, a) + alpha * {r(s, a) + gamma max{Q(s`, a`)} -  Q(s,a)}
	#               Q(s, a): 状態sにおける行動aを取った時のQ値      Q_s_a
	#               r(s, a): 状態sにおける報酬      r_s_a
	#               max{Q(s`, a`) 次の状態s`が取りうる行動a`の中で最大のQ値 mQ_s_a)
	Q_s_a = get_Qvalue(state,action,Q)#状態sにおける行動aを取った時のQ値  
	n_observation, reward, done, info = env.step(action)

	n_state = state_from_observation(n_observation) #次状態next_stateを取得
	r_s_a = reward #状態sで行動aを取った時の報酬R(s,a)

	# 次状態n_stateが取りうる行動n_actionの中で最大のQ値を求める
	mQ_ns_a = max([get_Qvalue(n_state,n_action,Q) for n_action in actions])

	# calculate
	q_value = Q_s_a + alpha * ( r_s_a +  gamma * mQ_ns_a - Q_s_a)

	# update
	Q[state,action] = q_value
	return n_observation, reward, done, info

def discrete(rangeMin,rangeMax,n,value):
	"""離散値に変換するメソッド"""
	interval = (rangeMax - rangeMin) / n
	r = rangeMin
	for i in range(n):
		r += interval
		if r > value:
			return i
	return i


if __name__ == '__main__':
	actions = [0,1]
	greedy_ratio = 0.9 # ε-greedy法でのε
	M = 10000

	# 行動価値関数。こいつを学習させる。
	# stateを離散的に定義するべき。
	# Q(s,a)は{(s,a):value,(s,a):value,..}というような形式。
	Q = {}
	for pos in range(DISCRETE['pos'][2]):
		for cart_vec in range(DISCRETE['cart_vec'][2]):
			for angle in range(DISCRETE['angle'][2]):
				for pole_vec in range(DISCRETE['pole_vec'][2]):
					for a in actions:
						Q[(pos,cart_vec,angle,pole_vec),a] = random.uniform(1,10)


	env = gym.make('CartPole-v0')
	total_reward_sum = np.zeros(M)


	for i_episode in range(M):
		# 学習
		observation = env.reset()
		for t in range(200):
			# env.render()

			# 状態からある方法で行動a_tを選択。ε-greedyを用いる。
			action = action_e_greedy(observation,greedy_ratio,actions,Q)

			# 選択した行動から、実際に行動を行い、Q(s,a)を更新
			observation, reward, done, info = action_and_update_Qvalue(observation,action,Q,env,actions,t)
			# print("pos:{0}, cart_vec:{1}, angle:{2}, pole_vec:{3}".format(observation[0],observation[1],observation[2],observation[3]))
			# print("state:{0}".format(state_from_observation(observation)))
			# print("reward:{0}, done:{1}".format(reward,done))

			if done:#t == 99:#done:
				print("Episode finished after {} timesteps".format(t+1))
				# print("total_reward is {0}".format(total_reward))
				# total_reward_sum[i_episode] = total_reward
				# if i_episode == M-1:
					# print("all total reward is {0}".format(total_reward_sum))
				break



	#方策決定
	print("方策決定")
	print(Q.values())
	observation = env.reset()
	while True:
		env.render()
		action = action_greedy(observation,actions,Q)
		observation, reward, done, info = env.step(action)




