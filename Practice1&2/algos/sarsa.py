import random
from collections import defaultdict
from env.gridworld_env import Action
import time
from tqdm import tqdm  

def sarsa(env, episodes=5000, alpha=0.1, gamma=0.99, epsilon=0.1, render=False, log_interval=100):
    # 낙관적인 초기값 (탐험 유도)
    Q = defaultdict(lambda: {a: 0.0 for a in Action})

    all_rewards = []
    success_count = 0

    for episode in tqdm(range(episodes), desc="Training SARSA"):
        state = tuple(env.reset())
        if random.random() < epsilon:
            action = random.choice(list(Action))
        else:
            action = max(Q[state], key=Q[state].get)

        done = False
        total_reward = 0

        while not done:
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
