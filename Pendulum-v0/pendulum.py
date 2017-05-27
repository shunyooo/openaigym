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
		self.env = gym.make("Pendulum-v0")


	def normalize(self, state):
		high = self.env.observation_space.high # 状態の最大値の配列
		low = self.env.observation_space.low   # 状態の最小値の配列
		return (state - low)/(high - low) # 全ての値を0-1に正規化 

	def getAction(self,sigma,mu,state,debug=False):
		"""行動決定"""
		max_a = 2.0; min_a = -2.0
		action = np.random.normal(np.dot(mu.T, state), sigma ** 2, 1)
		if debug == True:
			print("action:",action)
		action = min(action,max_a)
		action = max(action,min_a)
		return [action]

	def PolicyGradient(self,L,M,T,options):
		"""政策勾配アルゴリズム"""

		N = 4 #モデルパラメータ数(mu:3次元。状態数に基づく sigma:1次元)

		gamma = options.gamma
		alpha = options.alpha

		# 政策モデルパラメータをランダムに初期化
		mu = np.random.rand(N-1) - 0.5 #平均。ガウス分布の軸。
		sigma = np.random.rand() * 4  #分散。ガウス分布の幅。

		# 政策反復
		for l in range(L):
			dr = 0
			drs = np.empty((0),float)# 割引報酬和 1*M
			der = np.empty((0,N),float)# 勾配の和 M*N
			rewards = np.empty((0,T),float)# 報酬 M*T

			for m in range(M): # エピソード。標本抽出。

				# 配列drsのmエピソード目を動的確保
				drs = np.append(drs, 0)
				# 行列derのm行目を動的確保
				der = np.append(der,np.zeros((1,N)),axis=0)
				# 配列rewardsのm行目を確保
				rewards = np.append(rewards,np.zeros((1,T)),axis=0)


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

					# 平均muに関する勾配の観測. m行目の0行から-1行まで
					der[m,:-1] = der[m,:-1] + (((action-np.dot(mu.T, state))*state)/(sigma**2)).T

					# 標準偏差sigmaに関する勾配の観測.m行目の最後の要素
					der[m,-1] = der[m,-1] + ((action-np.dot(mu.T, state))**2-(sigma**2))/(sigma**3)

					# 割引報酬和の観測
					rewards[m,t] = reward
					drs[m] = drs[m] + (gamma**t)*rewards[m,t]# エピソード毎
					dr = dr + (gamma**t)*rewards[m,t]# 政策毎 デバッグ用


					if done:
						#print("Episode %d finished after {} timesteps".format(t+1) % m)
						break

			print("政策:{}".format(l))
			# 最小ベースラインを計算
			b = np.dot(drs,np.diag(np.dot(der,der.T)))/np.trace(np.dot(der,der.T))

			print("最小ベースライン:b")
			pprint(b)

			# 勾配を推定
			derJ = 1/M * (np.dot((drs-b),der)).T

			# モデルパラメータを更新
			mu = mu + alpha * derJ[:-1]
			sigma = sigma + alpha * derJ[-1]

			print("updated_mu")
			pprint(mu)
			print("updated_sigma")
			pprint(sigma)

			print("Max={:.2f}, Min={:.2f}, Avg={:.2f}".format(np.max(drs),np.min(drs),np.mean(drs)))
			print("-"*30)


if __name__ == '__main__':
	mc = Pendulum()

	L = 500
	M = 200
	T = 200
	options = AttributeDict({"gamma":0.99,"alpha":0.1})

	mc.PolicyGradient(L,M,T,options)