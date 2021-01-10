import psutil
import os
import sys

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
        print("eposide : {}, reward : {}".format(i, eposideReward))
    return eposideRewards

def expectSARSAPlay(env, train, render, agent, eposideNum):
    eposideRewards = []
    for i in range(eposideNum):
        state = env.reset()
        eposideReward = 0
        while True:
            print("代理内存占用:{}".format(sys.getsizeof(agent)))
            if render:
                env.render()
            action = agent.makeAction(state)
            nextState, reward, done, _ = env.step(action)
            eposideReward += reward
            if train:
                agent.learn(state, action, reward, nextState, done)
                printMemoryUsage()
            state = nextState
            if done:
                eposideRewards.append(eposideReward)
                break
        print("eposide : {}, reward : {}".format(i, eposideReward))
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

def printMemoryUsage():
    print("内存占用:{}".format(psutil.Process(os.getpid()).memory_info().rss))