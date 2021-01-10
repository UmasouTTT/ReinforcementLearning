import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from timeDifferencePlay import expectSARSAPlay
import psutil
import os
import sys

def printMemoryUsage():
    print("内存占用:{}".format(psutil.Process(os.getpid()).memory_info().rss))

class DQNReplayer:
    def __init__(self, capacity):
        self.memory = pd.DataFrame(index=range(capacity),
                                   columns=['observation', 'action', 'reward', 'next_observation', 'done'])
        self.i = 0
        self.count = 0
        self.capacity = capacity

    def store(self, *args):
        self.memory.loc[self.i] = args
        self.i = (self.i + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)


    def sample(self, size):
        indices = np.random.choice(self.count, size=size)
        return (np.stack(self.memory.loc[indices, field]) for field in self.memory.columns)

class DQNAgent:
    def __init__(self, env, net_kwargs={}, gamma=0.99, epsilon=0.01,
                 replayer_capacity=1, batch_size=64):
        observation_dim = env.observation_space.shape[0]
        self.action_n = env.action_space.n
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.replayer = DQNReplayer(replayer_capacity)
        self.evaluation_net = self.build_network(input_size=observation_dim,
                                                 output_size=self.action_n, **net_kwargs)
        self.target_net = self.build_network(input_size=observation_dim,
                                                 output_size=self.action_n, **net_kwargs)
        self.target_net.set_weights(self.evaluation_net.get_weights())

    def build_network(self, input_size, output_size, hidden_sizes,
                      activation=tf.nn.relu, output_action=None,
                      learning_rate=0.01):
        model = tf.keras.Sequential()

        for layer, hidden_size in enumerate(hidden_sizes):
            kwargs = dict(input_shape=(input_size,)) if not layer else {}
            model.add(tf.keras.layers.Dense(units=hidden_size,
                                            activation=activation,
                                            **kwargs))
        model.add(tf.keras.layers.Dense(units=output_size,
                                        activation=output_action,
                                        ))
        optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
        model.compile(loss='mse', optimizer=optimizer)
        return model

    def learn(self, observation, action, reward, next_observation, done):
        self.replayer.store(observation, action, reward, next_observation, done)
        observations, actions, rewards, next_observations, dones = self.replayer.sample(self.batch_size)
        next_qs = self.target_net.predict(next_observations)
        next_max_qs = next_qs.max(axis=-1)
        us = rewards + self.gamma * (1.-done) * next_max_qs
        targets = self.evaluation_net.predict(observations)
        tf.keras.backend.clear_session()
        targets[np.arange(us.shape[0]), actions] = us
        self.evaluation_net.fit(observations, targets, verbose=0)
        tf.keras.backend.clear_session()
        if done:
            self.target_net.set_weights(self.evaluation_net.get_weights())

    def makeAction(self, observation):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_n)
        qs = self.evaluation_net.predict(observation[np.newaxis])
        tf.keras.backend.clear_session()
        return np.argmax(qs)


net_kwargs = {'hidden_sizes': [64,], 'learning_rate': 0.01}
env = gym.make('MountainCar-v0')
agent = DQNAgent(env, net_kwargs=net_kwargs)
eposideRewards = expectSARSAPlay(env=env, train=True, render=False, agent=agent, eposideNum=5000)
plt.plot(eposideRewards)
plt.show()

