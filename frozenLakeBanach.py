import gym
import numpy as np

env = gym.make('FrozenLake-v0')
env = env.unwrapped
print(env.unwrapped.P[14][2])

def policyIterator(env):
    policy = np.ones((env.nS, env.nA)) / env.nA
    while True:
        stateValue = evalutePolicy(env, policy)
        if policyImprove(env, stateValue, policy):
            break
    return stateValue, policy

def stateValue2actionValue(stateValue, env, gamma = 1.0, state = None):
    actionVlue = np.zeros(env.nA)
    if None != state:
        actionValue = np.zeros(env.nA)
        for action in range(env.nA):
            for p, nextState, reward, done in env.P[state][action]:
                actionValue[action] += p * (reward + gamma * stateValue[nextState] * (1 - done))
    else:
        actionValue = np.zeros((env.nS, env.nA))
        for s in range(env.nS - 1):
            for action in range(env.nA):
                for p, nextState, reward, done in env.P[s][action]:
                    actionValue[s][action] += p * (reward + gamma * stateValue[nextState] * (1 - done))
    return actionValue

def evalutePolicy(env, policy, targetDelta = 1e-6):
    stateValue = np.zeros(env.nS)
    while True:
        delta = 0.
        for state in range(env.nS):
            newStateValue = sum(policy[state] * stateValue2actionValue(stateValue, env, state=state))
            delta = max(delta, abs(newStateValue - stateValue[state]))
            stateValue[state] = newStateValue
        if delta < targetDelta:
            break
    return stateValue

def policyImprove(env, newStateValue, policy):
    isSame = True
    for state in range(env.nS):
        bestAction = np.argmax(stateValue2actionValue(newStateValue, env, state=state))
        if 1 != policy[state][bestAction]:
            #policy[state] = np.zeros(env.nA)
            policy[state] = 0.
            policy[state][bestAction] = 1.
            isSame = False
    return isSame

def valueIterator(env, targetDelta = 1e-6):
    stateValue = np.zeros(env.nS)
    while True:
        delta = 0
        for state in range(env.nS):
            newStateValue = max(stateValue2actionValue(stateValue, env, state=state))
            delta = max(delta, abs(newStateValue - stateValue[state]))
            stateValue[state] = newStateValue
        if delta < targetDelta:
            print(delta)
            break
    policy = np.zeros((env.nS, env.nA))
    for state in range(env.nS):
        bestAction = np.argmax(stateValue2actionValue(stateValue, env, state=state))
        policy[state][bestAction] = 1.
    return stateValue, policy

print('策略迭代结果:')
stateValue, policy = policyIterator(env)
print('状态价值函数 = {}'.format(stateValue.reshape((4, 4))))
print('最优策略 = {}'.format(np.argmax(policy, axis=1).reshape(4, 4)))

print('价值迭代结果:')
stateValue, policy = valueIterator(env)
print('状态价值函数 = {}'.format(stateValue.reshape((4, 4))))
print('最优策略 = {}'.format(np.argmax(policy, axis=1).reshape(4, 4)))




