
"""MONTE CARLO, EPSILON - SOFT POLICY"""

# State s is discretised, one state [t1, t2, t1_d, t2_d] is represented as number (t1)*1000 + (t2)*100 + (t1_d)*10 + (t2_d)*1
# pi, policy is saved as a dictionary pi(s) = [ pi(a0|s), pi(a1|s), pi(a2|s) ], each 0 <= pi(ai, s) <= 1 and sum(pi(s)) = 1
# Q is also saved as a dictionary Q(s) = [ Q(s, a0), Q(s, a1), Q(s, a2) ], where each Q(s, a) is a rational number
# Returns is also saved as a dictionary R(s) = [ R(s, a0), R(s, a1), R(s, a2) ], where each R(s, a) is an empty list


import gym
import math
import random
from tqdm import tqdm


def getAngle(cos_val, sin_val):
    ang = math.atan2(sin_val, cos_val)
    return ang*180/math.pi + 180

def transformObs(obs):
    ang1 = int(getAngle(obs[0], obs[1])/60)
    ang2 = int(getAngle(obs[2], obs[3])/60)
    new_obs = []
    new_obs.append(ang1)
    new_obs.append(ang2)
    new_obs.append(int(round((obs[4] + 12.57)/5)))
    new_obs.append(int(round((obs[5] + 28.28)/10)))
    return new_obs

def transformState(s):
    return s[0]*1000 + s[1]*100 + s[2]*10 + s[3]*1

def InitialiseMC():

    pi = {} 
    Q = {}
    Returns = {}

    for d1 in range(0, 6):
        for d2 in range(0, 6):
            for d3 in range(0, 6):
                for d4 in range(0, 6):
                  s = d1*1000 + d2*100 + d3*10 + d4*1
                  p1 = random.uniform(0, 0.49)
                  p2 = random.uniform(0, 0.49)
                  pi[s] = [ p1, p2, 1-p1-p2 ]
                  Q[s] = [ random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1) ]
                  Returns[s] = [ [], [], [] ]

    return pi, Q, Returns


def Monte_Carlo_Control(env, episodes, gamma, epsilon):
    """On Policy first-visit MC control, Sutton-Barton, pg-101"""

    pi, Q, Returns = InitialiseMC()
    
    with open("pi_values_before.txt", 'w') as f:
        for x in pi:
            f.write(str(pi[x]))
            f.write('\n')

    for i in range(1, episodes+1):
        
        state = transformState(transformObs(env.reset()))
        history = [] # list containing (state-> action -> reward) history for an episode
        G = 0
        truncated = False
        # Generate 1 episode following pi
        step = 0
        while True:

            action = pi[state].index(max(pi[state]))
            stepData = env.step(action)

            observations = stepData[0]
            reward = stepData[1]
            terminated = stepData[2]
            truncated = stepData[3]
            info = stepData[4]

            history.append([state, action, reward])

            state = transformState(transformObs(observations))

            if terminated or truncated:
                break

        T = len(history)

        for t in range(T-1, -1, -1):
            G = gamma*G + history[t][2]
            St, At = [history[t][0], history[t][1]]
            for tt in range(0,t):
                if [St, At] == [history[tt][0], history[tt][1]]:
                    break
                
                Returns[St][At].append(G)
                Q[St][At] = sum(Returns[St][At])/len(Returns[St][At])
                A_optimal = (Q[St]).index(max(Q[St]))
                
                pi[St] = [epsilon/3, epsilon/3, epsilon/3]
                pi[St][A_optimal] = 1 - epsilon + epsilon/3
        print("Episode: ", i, "\tHistory: ", T)
        
    # print(Q)
    return pi, Q


env = gym.make("Acrobot-v1", new_step_api=True)

pi, Q = Monte_Carlo_Control(env, episodes = 1000, gamma = 1, epsilon = 1)


with open("pi_values_100.txt", 'w') as f:
    for x in pi:
        f.write(str(pi[x]))
        f.write('\n')

with open("Q_values_100.txt", 'w') as f:
    for x in Q:
        f.write(str(Q[x]))
        f.write('\n')


pi_optimal = {}

for s in Q:
    pi_optimal[s] = Q[s].index(max(Q[s]))


env = gym.make("Acrobot-v1",render = "human", new_step_api=True)
observation = env.reset(seed=42)

for _ in range(1000):
    observation = transformState(transformObs(observation))
    action = pi_optimal[observation]
    stepData = env.step(action)

    observation = stepData[0]
    reward = stepData[1]
    terminated = stepData[2]
    truncated = stepData[3]
    info = stepData[4]

    print("Observation : ", (observation))
    print("Reward : ", reward)
    print("Termnated : ", terminated)
    print("Truncated : ", truncated)
    print("Info : ", info)
    
    if terminated:
        observation = env.reset()

env.close()




