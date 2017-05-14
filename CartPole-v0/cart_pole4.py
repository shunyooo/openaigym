# 価値関数近似をやってみる
# 線形モデルを用いた線形近似モデルの例。
# cart_pole 
# 一旦保留。。

import gym
import time
import numpy as np
import random



# ガウス関数の中心行列(c1,c2,...,cB)T
C = []

# 関数の数
B = 10
weight:[Float] = np.zeros(B)



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
	max_q_value = -1 #最大の行動価値を保存
	for a in actions:#すごく単純な最大求めるやつ
		q_value = get_Qvalue(observation,a,Q)
		if q_value > max_q_value:
			best_actions = [a,]
			max_q_value = q_value
		elif q_value == max_q_value:
			best_actions.append(a)
	return random.choice(best_actions)#Q値の最大値が複数存在する場合はその中からランダムに選択。


def radical_func(observation,action,i,j):
	""" 線形近似モデルの基底関数。36個のガウス関数を用いる。
	関数ごとに中心ベクトルが違うので、それに注意。
	ガウス関数の中心を描く行動Actions = [0,1]ごとに
	4次元状態空間S上のグリッドのB個点に置く。
	actionがiに、中心ベクトルがjに対応。
	"""
	cart_pos,cart_vero,pole_angle,pole_vero = state_from_observation(observation)

	state = state_from_observation(observation)
	# 距離
	dist = np.linalg.norm(state-C[j])**2

	return I(action,i)*math.exp(dist/(2*(gamma**2)))

def I(action,i):
	return 1 if action == actions[i] else 0



def valueOfState(State):
	value = 0
	for i in range(B):




if __name__ == '__main__':
	env = gym.make('CartPole-v0')




	for i_episode in range(20):
		# reset関数でゲームが始まる。初期状態ランダム。戻り値は初期状態のobservation.
		e = 0
		observation = env.reset()
		for t in range(100):
			env.render()

			# action(行動):
			#	0:左へ押す
			#	1:右へ押す
			action = env.action_space.sample()

			# env.stepにactionを放り込むと、戻り値として色々返ってくる。
			# 1.環境を1step進める。もしepisodeの終わりに達すればreset()を自動的に呼び出す。
			# 2.引数としてactionオブジェクトを取り、戻り値としてobservation,reward,done,infoを含むタプルを返す。
			# 3.doneはboolean型でepisodeが終わったか否かを保持。
			# 4.infoはdictionary型でデバッグ情報など予備の診断情報を保持。
			# ---
			# observation(状態):
			#	0:カートの位置。-2.4~2.4
			# 	1:カートの速度。-Inf~Inf
			#	2:ポールの角度。-41.8~41.8
			#	3:ポールの先端速度。-Inf~Inf
			# ---
			# reward(報酬):
			# 	毎ステップ1が与えられる。
			# ---
			# done(終了判定):
			#	1.ポールの角度が ±20.9 を超えた時
			#	2.カートの位置が ±2.4 を超えた時
			# 	3.エピソードの長さが 200 を超えた時
			# ---
			observation, reward, done, info = env.step(action)

			if done:#ここで終了判定ができる。
				print("Episode finished after {} timesteps".format(t+1))
				break
			if i_episode == 0 and t == 0:
				# env.action_spaceはspaceクラスのオブジェクトで有効なactionを表している。
				print("action_space:",env.action_space)
				# env.observation_spaceはspaceクラスのオブジェクトで、有効なobservationを表す。
				# ここで出力すると、4次元のBoxクラスが出力される
				print("observation_space:",env.observation_space)
				# Boxクラスはhighとかlow属性を持つ
				print("env.observation_space.high is "),
				print(env.observation_space.high)
				print("env.observation_space.low is "),
				print(env.observation_space.low)