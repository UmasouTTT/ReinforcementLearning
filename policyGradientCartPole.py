import gym
import tensorflow as tf
import numpy as np
import pandas as pd
from tool import timeDifferencePlay
import matplotlib.pyplot as plt

#keras版本
class VPGAgent:
    def __init__(self, env, is_has_base_func, policy_kwargs={}, gamma=0.99):
        inputSize = env.observation_space.shape[0]
        self.action_num = env.action_space.n
        self.gamma = gamma
        self.is_has_base_func = is_has_base_func
        self.trace = []
        self.evaluation_net = self.makeNet(inputSize, outputSize=self.action_num, **policy_kwargs)
        if self.is_has_base_func:
            self.base_net = self.makeNet(inputSize, outputSize=1,
                                         output_activation=None,
                                         loss_func=tf.losses.mse,
                                         **policy_kwargs)

    def makeNet(self, inputSize, outputSize, hidden_sizes, activation=tf.nn.relu,
                output_activation=tf.nn.softmax, learning_rate=0.01, loss_func=tf.losses.categorical_crossentropy):
        model = tf.keras.Sequential()
        for layer, hiddenLayer in enumerate(hidden_sizes):
            kwags = dict(input_shape=(inputSize,)) if not layer else {}
            model.add(tf.keras.layers.Dense(units=hiddenLayer,
                                            activation=activation,
                                            **kwags))
        model.add(tf.keras.layers.Dense(units=outputSize,
                                        activation=output_activation))
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        model.compile(optimizer=optimizer, loss=loss_func)
        return model

    def makeAction(self, observation):
        prob = self.evaluation_net.predict(observation[np.newaxis])[0]
        return np.random.choice(self.action_num, p=prob)

    def learn(self, observation, action, reward, done):
        self.trace.append((observation, action, reward))

        if done:
            df = pd.DataFrame(self.trace, columns=['observation', 'action', 'reward'])
            df['discount'] = self.gamma ** df.index.to_series()
            df['discount_reward'] = df['reward'] * df['discount']
            df['discount_return'] = df['discount_reward'][::-1].cumsum()
            df['psi'] = df['discount_return']

            input = np.stack(df['observation'])
            if self.is_has_base_func:
                df['base_net'] = self.base_net.predict(input)
                # df['psi'] -= df['base_net']
                # base_func_output = df['psi'].values[:, np.newaxis]
                df['psi'] -= df['base_net'] * df['discount']
                df['return'] = df['discount_return'] / df['discount']
                base_func_output = df['return'].to_numpy()[:, np.newaxis]
                self.base_net.fit(input, base_func_output, verbose=0)

            output = np.eye(self.action_num)[df['action']]
            sampleWeight = df['psi'].values[:, np.newaxis]

            self.evaluation_net.fit(input, output,sample_weight=sampleWeight , verbose=0)

            self.trace.clear()




net_kwargs = {'hidden_sizes': [10,], 'learning_rate': 0.01}
env = gym.make('CartPole-v0')
agent = VPGAgent(env, is_has_base_func=True, policy_kwargs=net_kwargs)
eposideRewards = timeDifferencePlay.monteCarlo_play(env=env, render=False, agent=agent, eposideNum=300)
plt.plot(eposideRewards)
plt.show()






