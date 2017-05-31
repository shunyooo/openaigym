import gym
import numpy as np
from pprint import pprint
import decimal

class AttributeDict(object):
    """辞書をドット表記でアクセスできるように変換"""
    def __init__(self, obj):
        self.obj = obj

    def __getstate__(self):
        return self.obj.items()

    def __setstate__(self, items):
        if not hasattr(self, 'obj'):
            self.obj = {}
        for key, val in items:
            self.obj[key] = val

    def __getattr__(self, name):
        if name in self.obj:
            return self.obj.get(name)
        else:
            return None

    def fields(self):
        return self.obj

    def keys(self):
        return self.obj.keys()



class Pendulum:
	def __init__(self):
		self.env = gym.make("MountainCarContinuous-v0")
		# self.env = gym.make("Pendulum-v0")

	def normalize(self, state):
		high = self.env.observation_space.high # 状態の最大値の配列
		low = self.env.observation_space.low   # 状態の最小値の配列
		return (state - low)/(high - low) # 全ての値を0-1に正規化 

	def getAction(self,sigma,mu,state,debug=False):
		"""行動決定"""
		max_a = self.env.action_space.high[0]
		min_a = self.env.action_space.low[0]
		action = np.random.randn() * sigma + np.dot(mu.T, state) 
		#action = np.random.normal(np.dot(mu.T, state), sigma ** 2, 1)[0]
		if debug == True:
			print("sigma:",sigma)
			print("mu.T・state:",np.dot(mu.T, state))
			print("action:",action)
			print("-"*30)
		action = min(action,max_a)
		action = max(action,min_a)
		return [action]

	def NaturalActorCriric(self,L,M,T,options):
		"""自然勾配法アルゴリズム"""

		N = self.env.observation_space.shape[0]+1  #モデルパラメータ数(mu:3次元。状態数に基づく sigma:1次元)

		gamma = options.gamma
		alpha = options.alpha

		# 政策モデルパラメータをランダムに初期化
		mu = np.random.rand(N-1) - 0.5 #平均。ガウス分布の軸。
		sigma = np.random.rand() * (self.env.action_space.high[0] - self.env.action_space.low[0])/2 #分散。ガウス分布の片幅。
		#mu = np.array([0.4528,0.47625])
		#sigma = 1.7027


		# デザイン行列Z,報酬ベクトルqおよび
		# アドバンテージ関数のモデルパラメータwの初期化
		Z = np.zeros((M,N))
		q = np.zeros((M,1))
		w = np.zeros((N))

		# 政策反復
		for l in range(L):
			dr = 0
			rewards = np.empty((0,T),float)# 報酬 M*T
			goal_n = 0

			for m in range(M): # エピソード。標本抽出。
				# 行列derのm行目を動的確保
				der = np.zeros((N))
				# 配列rewardsのm行目を確保
				rewards = np.append(rewards,np.zeros((1,T)),axis=0)

				actions = np.zeros((T))
				states = np.zeros((N-1,T))
				# 状態の初期化
				state = self.normalize(self.env.reset())# 状態を初期化、0-1に正規化
				for t in range(T): # ステップ
					debug = False
					if m == M-1 and l%5 == 0:
						self.env.render()
						debug = True

					# 行動決定
					action = self.getAction(sigma,mu,state,debug)

					# 行動実行、観測
					observation, reward, done, _ = self.env.step(action)
					state = self.normalize(observation)

					states[:,t] = state
					actions[t] = action[0]
					# 割引報酬和の観測
					rewards[m,t] = reward # デバッグ用
					dr += (gamma**t)*rewards[m,t]# 政策毎

					if done:
						if t < T-2:
							#print("Episode %d finished after {} timesteps".format(t+1) % m)
							goal_n += 1
							break
						pass

				for t in range(T):
					# 平均muに関する勾配の計算
					der[:-1] += (actions[t] - np.dot(mu.T,states[:,t]))*states[:,t]/(sigma**2)
					# 標準偏差sigmaに関する勾配の計算
					der[-1] += ((actions[t]-np.dot(mu.T,states[:,t]))**2-(sigma**2))/(sigma**3)
					# デザイン行列Z及び報酬ベクトルq
					Z[m,:] += (gamma**t)*der
					q[m] += (gamma**t)*(rewards[m,t])

			# r - V(s1)
			q -= dr/M

			# 最小二乗法を用いてアドバンテージ関数のモデルパラメータを推定
			Z[:,-1] = np.ones((M))
			pinv = np.linalg.pinv(np.dot(Z.T,Z))
			w = np.dot(np.dot(pinv,Z.T),q).reshape(N,)

			# wを用いてモデルパラメータを更新
			mu += alpha * w[:-1]
			sigma += alpha * w[-1]

			print("政策:{}".format(l))
			print("updated_mu")
			pprint(mu)
			print("updated_sigma")
			pprint(sigma)
			print("w",w)
			print("Avg={:.2f}".format(dr/M))
			print("goal数:",goal_n,"/",M)
			print("-"*30)


if __name__ == '__main__':
	p = Pendulum()

	L = 1000
	M = 200
	T = 1000
	options = AttributeDict({"gamma":1.0,"alpha":0.01})

	p.NaturalActorCriric(L,M,T,options)