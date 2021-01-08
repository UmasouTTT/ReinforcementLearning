import gym
import numpy as np
import matplotlib.pyplot as plt
from timeDifferencePlay import SARSAPlay

env = gym.make('MountainCar-v0')

print('观测空间 = {}'.format(env.observation_space))
print('动作空间 = {}'.format(env.action_space))
print('位置范围 = {} - {}'.format(env.min_position, env.max_position))
print('速度范围 = {} - {}'.format(-env.max_speed, env.max_speed))
print('目标位置 = {}'.format(env.goal_position))


#恒定向右施力
def alwaysPressureRight(env):
    positions = []
    speeds = []
    observation = env.reset()
    while True:
        positions.append(observation[0])
        speeds.append(observation[1])
        observation, reward, done, _ = env.step(1)
        if done:
            break
    if observation[0] > 0.5:
        print("成功到达")
    else:
        print("失败")

    fig, ax = plt.subplots()
    ax.plot(positions, label="position")
    ax.plot(speeds, label="speed")
    ax.legend()
    plt.show()

class TileCoder:
    def __init__(self, layers, features):
        self.layers = layers
        self.features = features
        self.codebook = {}

    def get_feature(self, codeword):
        if codeword in self.codebook:
            return self.codebook[codeword]
        count = len(self.codebook)
        if count >= self.features:
            return hash(codeword) % self.features
        else:
            self.codebook[codeword] = count
        return count

    def __call__(self, floats=(), ints=()):
        dim = len(floats)
        scaled_floats = tuple(f * self.layers * self.layers for f in floats)
        features = []
        for layer in range(self.layers):
            codeword = (layer,) + tuple(int((f + (1 + dim * i) * layer)
                                            / self.layers) for i, f in enumerate(scaled_floats)) + ints
            feature = self.get_feature(codeword)
            features.append(feature)
        return features

#?
class SARSAAgent:
    def __init__(self, env, layers=8, features=1893, gamma=1.,
                 learning_rate=0.03, epsilon=0.001):
        self.action_n = env.action_space.n
        self.obs_low = env.observation_space.low
        self.obs_scale = env.observation_space.high - env.observation_space.low
        self.encoder = TileCoder(layers, features)
        self.w = np.zeros(features)
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon

    def encode(self, observation, action):
        states = tuple((observation - self.obs_low) / self.obs_scale)
        actions = (action,)
        return self.encoder(states, actions)

    def get_q(self, observation, action):
        features = self.encode(observation, action)
        return self.w[features].sum()

    def makeAction(self, observation):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_n)
        else:
            qs = [self.get_q(observation, action) for action in range(self.action_n)]
            return np.argmax(qs)

    def learn(self, observation, action, reward, next_observaion, done, next_action):
        u = reward + self.gamma * self.get_q(next_observaion, next_action) * (1. - done)
        td_error = u - self.get_q(observation, action)
        features = self.encode(observation, action)
        self.w[features] += (self.learning_rate * td_error)

agent = SARSAAgent(env)
eposideRewards = SARSAPlay(env=env, train=True, render=False, agent=agent, eposideNum=5000)
plt.plot(eposideRewards)
plt.show()



