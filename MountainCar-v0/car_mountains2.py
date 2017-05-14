import gym
from math import *
import numpy as np
import random
from pprint import *

ACTIONS = [0,1,2]
STATE_POSITIONS = [-1.2,-0.35,0.5]#位置
STATE_VEROCITYS = [-1.5,-0.5,0.5,1.5]#速度

M = 20
T = 200
GAMMA = 0.95 #割引率
SIGMA = 0.5  #ガウス関数の幅
EPSILON = 0.9

C = []#中心ベクトル C[(action,pos,vero)]でアクセス。
for action in ACTIONS:
	for vero in STATE_VEROCITYS:
		for pos in STATE_POSITIONS:
			C.append((action,pos,vero))
C = np.array(C, dtype=float)

# 用意する基底関数の数。
B = len(STATE_POSITIONS) * len(STATE_VEROCITYS) * len(ACTIONS)

# 重み。これを更新することが学習。
weight = np.zeros(B, dtype=float)

def printEnv():
	print("-"*40)
	print("行動Action:{0}個".format(len(ACTIONS)))
	pprint(ACTIONS)
	print("中心ベクトルC:{0}個".format(len(C)))
	pprint(C)
	print("基底関数の数:{0}".format(B))
	print("-"*40)


def get_action_e_greedy(observation):
	"""ε-greedy法で行動を決定"""
	if EPSILON > random.random():
		#ランダムに行動選択
		return random.choice(ACTIONS)
	else:
		#greedyに行動選択
		return get_action_greedy(observation)

def get_action_greedy(observation):
	"""Q(s,a)を比較してgreedy法で行動を決定。"""
	best_actions = []  #最高の行動が複数存在した場合
	max_q_value = -1 #最大の行動価値を保存
	for a in ACTIONS:#すごく単純な最大求めるやつ
		q_value = getQvalue(observation,a)
		if q_value > max_q_value:
			best_actions = [a,]
			max_q_value = q_value
		elif q_value == max_q_value:
			best_actions.append(a)
	return random.choice(best_actions)#Q値の最大値が複数存在する場合はその中からランダムに選択。


def radical_func(observation,action,i):
	""" 線形近似モデルの基底関数。B = 36個のガウス関数を用いる。
	関数ごとに中心ベクトルが違うので、それに注意。
	ガウス関数の中心を描く行動Actions = [-0.2,0,0.2]ごとに
	2次元状態空間S上のグリッド{-1.2,-0.35,0.5}*{-1.5,-0.5,0.5,1.5}
	の12点におく。中心ベクトルは12通りになる。
	actionがiに、中心ベクトルがjに対応。
	"""
	state = state_from_observation(observation)
	print(state,C[i])

	# 距離
	dist = np.linalg.norm(state-C[i][1:])**2
	return I(action,i)*exp(dist/(2*(GAMMA**2)))

def I(action,i):
	interval = len(STATE_POSITIONS) * len(STATE_VEROCITYS)
	for a_i,b in enumerate(range(0,B,interval)):
		if b > i:
			break
	a_i -= 1
	print(b,a_i)
	return 1 if action == ACTIONS[a_i] else 0

def getQvalue(observation,action):
	"""与えられた状態と行動における価値を返す。"""
	sumQ = 0
	state = state_from_observation(observation)
	for i in range(B):
		sumQ += radical_func(observation,action,i)*weight[i]
	return 	sumQ

def getQvalues(observation):
	"""与えられた状態における、全行動の行動価値をリストで返す。"""
	#{acrtion:value,..}
	return {action:getQvalue(observation,action) for action in ACTIONS}

def get_action(observation):
	#state = state_from_observation(observation)
	return random.choice(ACTIONS)

def state_from_observation(observation):
	return np.array(observation, dtype=float)
	pass

if __name__ == '__main__':
	printEnv()

	env = gym.make('MountainCar-v0')
	for i_episode in range(M):
		# reset関数でゲームが始まる。初期状態ランダム。戻り値は初期状態のobservation.
		observation = env.reset()
		for t in range(T):
			env.render()

			# 価値関数から、行動の決定
			action = get_action_e_greedy(observation)
			# 行動の実行。状態,報酬,終了判定などの取得。
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