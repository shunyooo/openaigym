import gym
env = gym.make('MountainCar-v0')
total = 0
N = 1000

observation = env.reset()
for t in range(N):
	env.render()
	print(observation)
	action = 2#env.action_space.contains(1)
	observation, reward, done, info = env.step(action)
	#print(reward)


# for i_episode in range(N):
# 	observation = env.reset()
# 	env.render()
# 	for t in range(200):
# 		action = 2 if observation[0] < -0.9 or observation[1] > 0 or (abs(observation[1]) < 0.001 and observation[0] < -0.5) else 0
# 		observation, reward, done, info = env.step(action)
# 		if done:
# 			print ("Episode %d finished after {} timesteps".format(t+1) % i_episode)
# 			total += t+1
# 			break
# print ("Average reward: %f" % (-total / N))
