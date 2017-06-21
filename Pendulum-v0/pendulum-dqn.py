#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy, sys
import numpy as np
from collections import deque

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Chain, optimizers, Variable, serializers


class Neuralnet(Chain):
    """ DQN用のNN
    100ユニットが３層。活性化関数はleaky_relu。

    Extends:
        Chain
    """

    def __init__(self, n_in, n_out):
        """init
        Arguments:
            n_in {[type]} -- 入力ユニット数。状態数に相当。
            n_out {[type]} -- 出力ユニット数。行動数に相当。
        """
        super(Neuralnet, self).__init__(
            L1 = L.Linear(n_in, 100),
            L2 = L.Linear(100, 100),
            L3 = L.Linear(100, 100),
            Q_value = L.Linear(100, n_out, initialW=np.zeros((n_out, 100), dtype=np.float32))
        )

    def Q_func(self, x):
        h = F.leaky_relu(self.L1(x))
        h = F.leaky_relu(self.L2(h))
        h = F.leaky_relu(self.L3(h))
        h = self.Q_value(h)
        return h

class Agent():
    """DQNのAgent。
    [description]
    """

    def __init__(self, n_st, n_act, seed):
        """init
        Arguments:
            n_st {[type]} -- 状態数
            n_act {[type]} -- 行動数
            seed {[type]} -- 乱数用のseed
        """
        np.random.seed(seed)
        sys.setrecursionlimit(10000)
        self.n_act = n_act
        self.model = Neuralnet(n_st, n_act)
        self.target_model = copy.deepcopy(self.model)
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)
        self.memory = deque()
        self.loss = 0
        self.step = 0
        self.gamma = 0.99
        self.mem_size = 1000
        self.batch_size = 100
        self.epsilon = 1
        self.epsilon_decay = 0.005
        self.epsilon_min = 0
        self.exploration = 1000
        self.train_freq = 10
        self.target_update_freq = 20

    def stock_experience(self, st, act, r, st_dash, ep_end):
        """[summary]
        
        [description]
        
        Arguments:
            st {[type]} -- [description]
            act {[type]} -- [description]
            r {[type]} -- [description]
            st_dash {[type]} -- [description]
            ep_end {[type]} -- [description]
        """
        self.memory.append((st, act, r, st_dash, ep_end))
        if len(self.memory) > self.mem_size:
            self.memory.popleft()

    def forward(self, st, act, r, st_dash, ep_end):
        s = Variable(st)
        s_dash = Variable(st_dash)
        Q = self.model.Q_func(s)
        tmp = self.target_model.Q_func(s_dash)
        tmp = list(map(np.max, tmp.data))
        max_Q_dash = np.asanyarray(tmp, dtype=np.float32)
        target = np.asanyarray(copy.deepcopy(Q.data), dtype=np.float32)
        for i in range(self.batch_size):
            target[i, act[i]] = r[i] + (self.gamma * max_Q_dash[i]) * (not ep_end[i])
        loss = F.mean_squared_error(Q, Variable(target))
        self.loss = loss.data
        return loss

    def suffle_memory(self):
        mem = np.array(self.memory)
        return np.random.permutation(mem)

    def parse_batch(self, batch):
        st, act, r, st_dash, ep_end = [], [], [], [], []
        for i in range(self.batch_size):
            st.append(batch[i][0])
            act.append(batch[i][1])
            r.append(batch[i][2])
            st_dash.append(batch[i][3])
            ep_end.append(batch[i][4])
        st = np.array(st, dtype=np.float32)
        act = np.array(act, dtype=np.int8)
        r = np.array(r, dtype=np.float32)
        st_dash = np.array(st_dash, dtype=np.float32)
        ep_end = np.array(ep_end, dtype=np.bool)
        return st, act, r, st_dash, ep_end

    def experience_replay(self):
        mem = self.suffle_memory()
        perm = np.array(range(len(mem)))
        for start in perm[::self.batch_size]:
            index = perm[start:start+self.batch_size]
            batch = mem[index]
            st, act, r, st_d, ep_end = self.parse_batch(batch)
            self.model.zerograds()
            loss = self.forward(st, act, r, st_d, ep_end)
            loss.backward()
            self.optimizer.update()

    def get_action(self, st):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.n_act), 0
        else:
            s = Variable(st)
            Q = self.model.Q_func(s)
            Q = Q.data[0]
            a = np.argmax(Q)
            return np.asarray(a, dtype=np.int8), max(Q)

    def reduce_epsilon(self):
        if self.epsilon > self.epsilon_min and self.exploration < self.step:
            self.epsilon -= self.epsilon_decay

    def train(self):
        if len(self.memory) >= self.mem_size:
            if self.step % self.train_freq == 0:
                self.experience_replay()
                self.reduce_epsilon()
            if self.step % self.target_update_freq == 0:
                self.target_model = copy.deepcopy(self.model)
        self.step += 1

    def save_model(self, model_dir):
        serializers.save_npz(model_dir + "model.npz", self.model)

    def load_model(self, model_dir):
        serializers.load_npz(model_dir + "model.npz", self.model)

import gym, sys
import numpy as np

def main(env_name, render=False, monitor=True, load=False, seed=0):

    env = gym.make(env_name)
    model_path = "./model/" + env_name + "_"

    n_st = env.observation_space.shape[0]
    if type(env.action_space) == gym.spaces.discrete.Discrete:
        # CartPole-v0, Acrobot-v0, MountainCar-v0
        n_act = env.action_space.n
        action_list = range(0, n_act)
    elif type(env.action_space) == gym.spaces.box.Box:
        # Pendulum-v0
        action_list = [np.array([a]) for a in [-2.0, 2.0]]
        n_act = len(action_list)

    agent = Agent(n_st, n_act, seed)
    if load:
        agent.load_model(model_path)

    print("エピソード\t報酬和\tepsilon\t誤差\t価値\tステップ")
    for i_episode in range(1000):
        observation = env.reset()
        r_sum = 0
        q_list = []
        for t in range(200):
            render = True if i_episode%10 == 0 else False

            if render:
                env.render()
            state = observation.astype(np.float32).reshape((1,n_st))
            act_i, q = agent.get_action(state)
            q_list.append(q)
            action = action_list[act_i]
            observation, reward, ep_end, _ = env.step(action)
            state_dash = observation.astype(np.float32).reshape((1,n_st))
            agent.stock_experience(state, act_i, reward, state_dash, ep_end)
            agent.train()
            r_sum += reward
            if ep_end:
                break
        print ("\t".join(map(str,[i_episode, r_sum, agent.epsilon, agent.loss, sum(q_list)/float(t+1) ,agent.step])))
        agent.save_model(model_path)

if __name__=="__main__":
    env_name = "Pendulum-v0"
    main(env_name)


