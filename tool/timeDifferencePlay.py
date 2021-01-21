import psutil
import os
import sys
import psutil

def SARSAPlay(env, train, render, agent, eposideNum):
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
                eposideRewards.append(eposideReward)
                break
        print("eposide : {}, reward : {}, memory left : {}".format(i, eposideReward, psutil.virtual_memory().available))
    return eposideRewards

def expectSARSAPlay(env, train, render, agent, eposideNum):
    eposideRewards = []
    for i in range(eposideNum):
        state = env.reset()
        eposideReward = 0
        step = 0
        while True:
            if render:
                env.render()
            action = agent.makeAction(state)
            nextState, reward, done, _ = env.step(action)
            eposideReward += reward
            if train:
                agent.learn(state, action, reward, nextState, done)
                if psutil.virtual_memory().available < 10000:
                    break
            state = nextState
            step += 1
            if done:
                eposideRewards.append(eposideReward)
                break
        print("eposide : {}, reward : {}, memory left : {}".format(i, eposideReward, psutil.virtual_memory().available))
    return eposideRewards

def SARSALambdaPlay(env, train, render, agent, eposideNum):
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


def monteCarlo_play(env, agent, render, eposideNum):
    eposideRewards = []
    for i in range(eposideNum):
        observation = env.reset()
        eposideReward = 0
        while True:
            if render:
                env.render()
            action = agent.makeAction(observation)
            nextState, reward, done, _ = env.step(action)
            eposideReward += reward
            agent.learn(observation, action, reward, done)
            observation = nextState
            if done:
                eposideRewards.append(eposideReward)
                break
        print("eposide : {}, reward : {}".format(i, eposideReward))
    return eposideRewards

def diffPolicyMonteCarlo_play(env, agent, playAgent, render, eposideNum):
    eposideRewards = []
    for i in range(eposideNum):
        observation = env.reset()
        eposideReward = 0
        stateActionRewardList = []
        while True:
            if render:
                env.render()
            action, prob = playAgent.makeAction()
            nextState, reward, done, _ = env.step(action)
            stateActionRewardList.append((observation, action, reward, prob))
            eposideReward += reward
            observation = nextState
            if done:
                agent.learn(stateActionRewardList)
                eposideRewards.append(eposideReward)
                break
        print("eposide : {}, reward : {}".format(i, eposideReward))
    return eposideRewards

def printMemoryUsage():
    print("内存占用:{}".format(psutil.Process(os.getpid()).memory_info().rss))