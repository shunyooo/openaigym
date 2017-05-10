# cart_pole 基本的な実行で探る

import gym
env = gym.make('CartPole-v0')
for i_episode in range(20):
	# reset関数でゲームが始まる。初期状態ランダム。戻り値は初期状態のobservation.
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