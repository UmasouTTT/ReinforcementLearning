from DQNMountainCar import DQNAgent
import numpy as np
import gym
from tool.timeDifferencePlay import expectSARSAPlay
import matplotlib.pyplot as plt

class DualDQNAgent(DQNAgent):
    def train_target_net(self, observations, targets):
        self.target_net.fit(observations, targets, verbose=0)

    def learn(self, observation, action, reward, next_observation, done):
        self.replayer.store(observation, action, reward, next_observation, done)
        observations, actions, rewards, next_observations, dones = self.replayer.sample(self.batch_size)
        if 0 == np.random.randint(0, 2):
            SAValues = self.evaluation_net_predict(observations)
            nextStateActionValues = self.evaluation_net_predict(next_observations)
            predictActions = np.argmax(nextStateActionValues, axis=-1)
            targetSAValues = self.target_net_predict(next_observations)
            us = rewards + self.gamma * targetSAValues[np.arange(targetSAValues.shape[0]), predictActions] * (1 - dones)
            SAValues[np.arange(SAValues.shape[0]), actions] = us
            self.train_evaluation_net(observations, SAValues)
        else:
            SAValues = self.target_net_predict(observations)
            nextSAValues = self.target_net_predict(next_observations)
            nextActions = np.argmax(nextSAValues, axis=-1)
            targetSAValues = self.evaluation_net_predict(next_observations)
            us = rewards + self.gamma * targetSAValues[np.arange(targetSAValues.shape[0]), nextActions] * (1 - dones)
            SAValues[np.arange(SAValues.shape[0]), actions] = us
            self.train_evaluation_net(observations, SAValues)

net_kwargs = {'hidden_sizes': [64,], 'learning_rate': 0.01}
env = gym.make('MountainCar-v0')
agent = DualDQNAgent(env, net_kwargs=net_kwargs)
eposideRewards = expectSARSAPlay(env=env, train=True, render=False, agent=agent, eposideNum=5000)
plt.plot(eposideRewards)
plt.show()

