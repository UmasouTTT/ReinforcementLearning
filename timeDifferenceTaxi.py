import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('Taxi-v3')
state = env.reset()
taxirow, taxicol, passloc, destidx = env.unwrapped.decode(state)
print("出租车位置 = {}".format((taxirow, taxicol)))
print("乘客位置 = {}".format(passloc))
print("目标位置 = {}".format(destidx))
env.render()
env.step(1)

#SARSA算法
class SARSAAgent:
    def __init__(self, env, gamma=0.9, learning_rate=0.1, epsilon=.01):
        self.env = env
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.action_n = env.action_space.n
        self.q = np.zeros((env.observation_space.n, env.action_space.n))

    def learn(self,  state1, action1, reward, state2, action2, done):
        u = reward + self.gamma * self.q[state2][action2] * (1. - done)
        self.q[state1][action1] += self.learning_rate * (u - self.q[state1][action1])

    def makeAction(self, state):
        if np.random.uniform() > self.epsilon:
            return np.argmax(self.q[state])
        else:
            return np.random.randint(env.action_space.n)


class expectSARSAAgent(SARSAAgent):

    def learn(self,  state1, action1, reward, state2, done):
        u = reward + self.gamma * (np.sum(self.q[state2]) * self.epsilon +
                                   np.max(self.q[state2]) * (1 - self.epsilon)) * (1. - done)
        self.q[state1][action1] += self.learning_rate * (u - self.q[state1][action1])

class QLearningAgent(SARSAAgent):
    def learn(self,  state1, action1, reward, state2, done):
        u = reward + self.gamma * (np.max(self.q[state2])) * (1. - done)
        self.q[state1][action1] += self.learning_rate * (u - self.q[state1][action1])

class doubleQLearningAgent(SARSAAgent):
    def __init__(self, env, gamma=0.9, learning_rate=0.1, epsilon=.01):
        super(doubleQLearningAgent, self).__init__(env, gamma, learning_rate, epsilon)
        self._q = np.zeros_like(self.q)

    def makeAction(self, state):
        temp_q = self.q + self._q
        if np.random.uniform() > self.epsilon:
            return np.argmax(temp_q[state])
        else:
            return np.random.choice(self.env.action_space.n)

    def learn(self,  state1, action1, reward, state2,  done):
        u = 0
        if 0 == np.random.randint(0, 2):
            actionChoosenByQ1 = np.argmax(self.q[state2])
            u = reward + self.gamma * self._q[state2][actionChoosenByQ1] * (1. - done)
            self.q[state1][action1] += self.learning_rate * (u - self.q[state1][action1])
        else:
            actionChoosenByQ1 = np.argmax(self._q[state2])
            u = reward + self.gamma * self.q[state2][actionChoosenByQ1] * (1. - done)
            self._q[state1][action1] += self.learning_rate * (u - self._q[state1][action1])


class SARSALambdaAgent(SARSAAgent):
    def __init__(self, env, lambd=0.9, beta=1., gamma=0.9, learning_rate=0.1, epsilon=.01):
        super(SARSALambdaAgent, self).__init__(env, gamma, learning_rate, epsilon)
        self.lambd = lambd
        self.beta = beta
        self.e = np.zeros_like(self.q)

    def clearE(self):
        self.e.fill(0.)

    def learn(self,  state1, action1, reward, state2, action2, done):
        self.e *= self.lambd * self.gamma
        self.e[state1][action1] = 1. + self.beta * self.e[state1][action1]
        u = reward + self.gamma * self.q[state2][action2] * (1. - done)
        self.q += self.learning_rate * self.e * (u - self.q)





def SARSAPlay(train, render, agent, eposideNum):
    eposideRewards = []
    for i in range(eposideNum):
        print("eposide : {}".format(i))
        state = env.reset()
        action = agent.makeAction(state)
        eposideReward = 0
        while True:
            if render:
                env.render()
            nextState, reward, done, _ = env.step(action)
            eposideReward += reward
            nextAction = agent.makeAction(nextState)
            if train:
                agent.learn(state, action, reward, nextState, nextAction, done)
            state, action = nextState, nextAction
            if done:
                eposideRewards.append(eposideReward)
                break
    return eposideRewards

def expectSARSAPlay(train, render, agent, eposideNum):
    eposideRewards = []
    for i in range(eposideNum):
        print("eposide : {}".format(i))
        state = env.reset()
        eposideReward = 0
        while True:
            if render:
                env.render()
            action = agent.makeAction(state)
            nextState, reward, done, _ = env.step(action)
            eposideReward += reward
            if train:
                agent.learn(state, action, reward, nextState, done)
            state = nextState
            if done:
                eposideRewards.append(eposideReward)
                break
    return eposideRewards

def SARSALambdaPlay(train, render, agent, eposideNum):
    eposideRewards = []
    for i in range(eposideNum):
        state = env.reset()
        action = agent.makeAction(state)
        eposideReward = 0
        while True:
            if render:
                env.render()
            nextState, reward, done, _ = env.step(action)
            eposideReward += reward
            nextAction = agent.makeAction(nextState)
            if train:
                agent.learn(state, action, reward, nextState, nextAction, done)
            state, action = nextState, nextAction
            if done:
                agent.clearE()
                eposideRewards.append(eposideReward)
                break
        print("eposide : {}, reward : {}".format(i, eposideReward))
    return eposideRewards






#测试SARSA
# agent = SARSAAgent(env)
# eposideRewards = SARSAPlay(train=True, render=False, agent=agent, eposideNum=5000)
# plt.plot(eposideRewards)
# plt.show()

#测试期望SARSA
# agent = expectSARSAAgent(env)
# eposideRewards = expectSARSAPlay(train=True, render=False, agent=agent, eposideNum=5000)
# plt.plot(eposideRewards)
# plt.show()

#测试QLearning
# agent = QLearningAgent(env)
# eposideRewards = expectSARSAPlay(train=True, render=False, agent=agent, eposideNum=5000)
# plt.plot(eposideRewards)
# plt.show()

#测试双重QLearning
# agent = doubleQLearningAgent(env)
# eposideRewards = expectSARSAPlay(train=True, render=False, agent=agent, eposideNum=5000)
# plt.plot(eposideRewards)
# plt.show()
#
# #测试LambdaSARSA
# agent = SARSALambdaAgent(env)
# eposideRewards = SARSALambdaPlay(train=True, render=False, agent=agent, eposideNum=5000)
# plt.plot(eposideRewards)
# plt.show()






