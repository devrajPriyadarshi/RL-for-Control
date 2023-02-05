"""MONTE CARLO, EPSILON - SOFT POLICY"""

# State s is discretised, one state [t1, t2, t1_d, t2_d] is represented as number (t1)*1000 + (t2)*100 + (t1_d)*10 + (t2_d)*1
# pi, policy is saved as a dictionary pi(s) = [ pi(a0|s), pi(a1|s), pi(a2|s) ], each 0 <= pi(ai, s) <= 1 and sum(pi(s)) = 1
# Q is also saved as a dictionary Q(s) = [ Q(s, a0), Q(s, a1), Q(s, a2) ], where each Q(s, a) is a rational number
# Returns is also saved as a dictionary R(s) = [ R(s, a0), R(s, a1), R(s, a2) ], where each R(s, a) is an empty list


import gym
import math
import random

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
                  pi[s] = [ p1, 1-p1-p2, p2 ]
                  Q[s] = [ 0, 0, 0 ]
                  Returns[s] = [ [], [], [] ]

    return pi, Q, Returns


def Monte_Carlo_Control(env, episodes, gamma, epsilon):
    """On Policy first-visit MC control, Sutton-Barton, pg-101"""

    pi, Q, Returns = InitialiseMC()

    i = 1
    while True:
        
        epsilon = 1/i
        state = transformState(transformObs(env.reset(seed = i)))
        history = [] # list containing (state-> action -> reward) history for an episode
        G = 0
        truncated = False
        # Generate 1 episode following pi
        for _ in range(500):

            action = pi[state].index(max(pi[state]))
            stepData = env.step(action)

            observations = stepData[0]
            reward = stepData[1]
            terminated = stepData[2]
            truncated = stepData[3]
            info = stepData[4]

            history.append([state, action, reward])

            state = transformState(transformObs(observations))

            if terminated:
                break

        T = len(history)
        print("Episode: ", i)
        print("History: ", T)
        
        i = i+1
        if i>episodes:
            break

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
    # print(Q)
    return pi, Q


env = gym.make("Acrobot-v1", new_step_api=True)

pi, Q = Monte_Carlo_Control(env, episodes = 500, gamma = 1, epsilon = 1)

with open("pi_values.txt", 'w') as f:
    for x in pi:
        f.write(str(pi[x]))
        f.write('\n')

with open("Q_values.txt", 'w') as f:
    for x in pi:
        f.write(str(Q[x]))
        f.write('\n')
