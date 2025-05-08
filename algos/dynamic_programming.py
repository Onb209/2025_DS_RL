# algos/dynamic_programming.py

import random
from env.gridworld_env import Action

def value_iteration(mdp, gamma=0.99, theta=1e-4):
    V = {s: 0 for s in mdp.states}
    while True:
        delta = 0
        for s in mdp.states:
            max_v = float('-inf')
            for a in mdp.actions:
                next_s, r, done = mdp.get_transition(s, a)
                v = r + gamma * (0 if done else V[next_s])
                max_v = max(max_v, v)
            delta = max(delta, abs(V[s] - max_v))
            V[s] = max_v
        if delta < theta:
            break

    pi = {}
    for s in mdp.states:
        best_a = None
        best_v = float('-inf')
        for a in mdp.actions:
            next_s, r, done = mdp.get_transition(s, a)
            v = r + gamma * (0 if done else V[next_s])
            if v > best_v:
                best_v = v
                best_a = a
        pi[s] = best_a
    return V, pi

def policy_iteration(mdp, gamma=0.99, theta=1e-4):
    pi = {s: random.choice(mdp.actions) for s in mdp.states}
    V = {s: 0 for s in mdp.states}

    while True:
        # 평가
        while True:
            delta = 0
            for s in mdp.states:
                a = pi[s]
                next_s, r, done = mdp.get_transition(s, a)
                v = r + gamma * (0 if done else V[next_s])
                delta = max(delta, abs(V[s] - v))
                V[s] = v
            if delta < theta:
                break

        # 개선
        policy_stable = True
        for s in mdp.states:
            old_action = pi[s]
            best_a = None
            best_v = float('-inf')
            for a in mdp.actions:
                next_s, r, done = mdp.get_transition(s, a)
                v = r + gamma * (0 if done else V[next_s])
                if v > best_v:
                    best_v = v
                    best_a = a
            pi[s] = best_a
            if best_a != old_action:
                policy_stable = False

        if policy_stable:
            break

    return V, pi
