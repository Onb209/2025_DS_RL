import random
from collections import defaultdict
from env.gridworld_env import Action
import time
from tqdm import tqdm  

epsilon_start = 1.0
# epsilon_start = 0.5
epsilon_end = 0.05
epsilon_decay = 0.98
# epsilon_start = 1.0
# epsilon_end = 0.05
# epsilon_decay = 0.8

#19: 
# epsilon_start = 0.5
# epsilon_end = 0.05
# epsilon_decay = 0.98
# gamma=0.95
# alpha = 0.3
# Q = defaultdict(lambda: {a: 1.0 for a in Action})
# 15: 
# epsilon_start = 1.0
# epsilon_end = 0.05
# epsilon_decay = 0.98 or 0.9
# gamma=0.95
# alpha = 0.3 or 0.1
# Q = defaultdict(lambda: {a: 1.0 for a in Action})


def sarsa(env, episodes=500, alpha=0.1, gamma=0.95, render=False, log_interval=100):
    epsilon = epsilon_start
    # 낙관적인 초기값 (탐험 유도)
    Q = defaultdict(lambda: {a: 1.0 for a in Action})

    all_rewards = []
    success_count = 0
    max_steps = 500

    for episode in tqdm(range(episodes), desc="Training SARSA"):
        state = tuple(env.reset())
        if random.random() < epsilon:
            action = random.choice(list(Action))
        else:
            action = max(Q[state], key=Q[state].get)

        done = False
        total_reward = 0
        steps = 0

        # while not done:
        while not done and steps < max_steps:
            if render and episode % log_interval == 0:
                env.render()
                time.sleep(0.05)

            next_state, reward, done = env.step(action.value)
            next_state = tuple(next_state)
            total_reward += reward

            if random.random() < epsilon:
                next_action = random.choice(list(Action))
            else:
                next_action = max(Q[next_state], key=Q[next_state].get)

            # SARSA update
            Q[state][action] += alpha * (
                reward + gamma * Q[next_state][next_action] - Q[state][action]
            )

            state, action = next_state, next_action

            steps += 1
        if epsilon > epsilon_end:
            epsilon *= epsilon_decay
        all_rewards.append(total_reward)
        if reward == 100:  # goal 도달 시
            success_count += 1

        # 로그 출력
        if (episode + 1) % log_interval == 0:
            avg_reward = sum(all_rewards[-log_interval:]) / log_interval
            success_rate = success_count / log_interval * 100
            print(f"[Episode {episode+1}] Avg Reward: {avg_reward:.2f}, Success Rate: {success_rate:.1f}%")
            success_count = 0  # 성공률 초기화

    # 최종 정책 반환
    policy = {state: max(actions, key=actions.get) for state, actions in Q.items()}
    return Q, policy
