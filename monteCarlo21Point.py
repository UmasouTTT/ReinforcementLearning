import gym
import numpy as np
import matplotlib.pyplot as plt
import time

env = gym.make('Blackjack-v0')
observation = env.reset()
print('观测= {}, 玩家 = {}, 庄家 = {}'.format(observation, env.player, env.dealer))

def plot(data):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    titles = ['without ace', 'with ace']
    have_aces = [0, 1]
    extent = [12, 22, 1, 11]
    for title, have_ace, axis in zip(titles, have_aces, axes):
        dat = data[extent[0]:extent[1], extent[2]:extent[3], have_ace].T
        axis.imshow(dat, extent=extent, origin='lower')
        axis.set_xlabel('player sum')
        axis.set_ylabel('dealer showing')
        axis.set_title(title)
    plt.show()

def observation2state(observation):
    return (observation[0], observation[1], int(observation[2]))

#同策略

def evaluate_action_monteCarlo_same_policy(env, policy, eposideNum):
    p = np.zeros_like(policy)
    c = np.zeros_like(policy)
    for i in range(eposideNum):
        print(i)
        actionStateDict = []
        observation = env.reset()
        g = 0
        while True:
            state = observation2state(observation)
            action = np.random.choice(env.action_space.n, p=policy[state])
            actionStateDict.append((state, action))
            observation, reward, done, _ = env.step(action)
            if done:
                g = reward
                break
        for state, action in actionStateDict:
            c[state][action] += 1
            p[state][action] = p[state][action] + (g - p[state][action]) / c[state][action]
    return p

def same_policy_start_explore(env, policy, eposideNum):
    p = np.zeros_like(policy)
    c = np.zeros_like(policy)
    for i in range(eposideNum):
        print('波次{}'.format(i))
        state = (np.random.randint(12, 22),
                 np.random.randint(1, 11),
                 np.random.randint(2))
        action = np.random.randint(2)
        env.reset()
        if 1 == state[2]:
            env.player = [1, state[0] - 11]
        else:
            if 21 == state[0]:
                env.player = [10, 9, 2]
            else:
                env.player = [10, state[0] - 10]
        env.dealer[0] = state[1]
        stateActionDict = []
        g = 0
        while True:
            stateActionDict.append((state, action))
            nextState, reward, done, _ = env.step(action)
            if done:
                g = reward
                break
            state = observation2state(nextState)
            action = np.random.choice(env.action_space.n, p=policy[state])
        for state, action in stateActionDict:
            c[state][action] += 1.
            p[state][action] += (g - p[state][action]) / c[state][action]
            policy[state] = 0.
            policy[state][np.argmax(p[state])] = 1.
    return p, policy

def same_policy_flexible(env, policy, eposideNum, eposilon = 0.1):
    p = np.zeros_like(policy)
    c = np.zeros_like(policy)
    for i in range(eposideNum):
        print(i)
        actionStateDict = []
        observation = env.reset()
        g = 0
        while True:
            state = observation2state(observation)
            action = np.random.choice(env.action_space.n, p=policy[state])
            actionStateDict.append((state, action))
            observation, reward, done, _ = env.step(action)
            if done:
                g = reward
                break
        for state, action in actionStateDict:
            c[state][action] += 1
            p[state][action] = p[state][action] + (g - p[state][action]) / c[state][action]
            policy[state] = eposilon / 2
            policy[state][np.argmax(p[state])] += 1 - eposilon
    return p

#异策略
def evaluate_action_monteCarlo_diff_policy(env, policy, actionPolicy, eposideNum):
    p = np.zeros_like(policy)
    c = np.zeros_like(policy)
    for i in range(eposideNum):
        print(i)
        actionStateDict = []
        observation = env.reset()
        g = 0
        importanceP = 1
        while True:
            state = observation2state(observation)
            action = np.random.choice(env.action_space.n, p=actionPolicy[state])
            actionStateDict.append((state, action))
            observation, reward, done, _ = env.step(action)
            if done:
                g = reward
                break
        for state, action in reversed(actionStateDict):
            c[state][action] += importanceP
            p[state][action] += (g - p[state][action]) * importanceP / c[state][action]
            importanceP *= policy[state][action] / actionPolicy[state][actionPolicy]
            if 0 == importanceP:
                break
    return p


#测试同策略
# policy = np.zeros((22, 11, 2, 2))
# policy[20:, :, :, 0] = 1
# policy[:20, :, :, 1] = 1
# q = evaluate_action_monteCarlo_same_policy(env, policy, 500000)
# v = (q * policy).sum(axis=-1)

#测试起始探索同策略
# policy = np.zeros((22, 11, 2, 2))
# policy[:, :, :, 1] = 1.
# q, policy = same_policy_start_explore(env, policy, 500000)
# v = q.max(axis=-1)
# plot(policy.argmax(-1))
# plot(v)

#测试柔性策略
policy = np.ones((22, 11, 2, 2)) * 0.5
q, policy = same_policy_flexible(env, policy, 500000)
v = q.max(axis=-1)
plot(policy.argmax(-1))
plot(v)








