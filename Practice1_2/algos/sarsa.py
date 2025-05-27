import random
from collections import defaultdict
from env.gridworld_env import Action
import time
from tqdm import tqdm  
from algos.dynamic_programming import plot_value_and_policy
import matplotlib.pyplot as plt

def sarsa(env, episodes=500, alpha=0.1, gamma=0.99, epsilon = 0.05, render=False, log_interval=100):
    # Q-value table 초기화: 모든 state-action 쌍에 대해 Q값을 0으로 초기화
    Q = defaultdict(lambda: {a: 0.0 for a in Action})

    all_rewards = []
    success_count = 0
    max_steps = 500

    for episode in tqdm(range(episodes), desc="Training SARSA"):
        state = tuple(env.reset())

        # epcsilon greedy로 action 선택
        if random.random() < epsilon:
            action = random.choice(list(Action)) # Exploration
        else:
            action = max(Q[state], key=Q[state].get) # Exploitation. 현재 Q value 기반 optimal action

        done = False
        total_reward = 0
        steps = 0

        # while not done:
        while not done and steps < max_steps:
            if render and episode % log_interval == 0:
                env.render()
                time.sleep(0.05)
            
            # action 후 next state와 reward 얻음
            next_state, reward, done = env.step(action.value) 
            next_state = tuple(next_state)

            total_reward += reward

            # next action 미리 선택
            if random.random() < epsilon:
                next_action = random.choice(list(Action))
            else:
                next_action = max(Q[next_state], key=Q[next_state].get)

            # SARSA update
            # next state에서 ε-greedy로 선택된 action의 Q값
            td_target = reward + gamma * Q[next_state][next_action] 
            Q[state][action] += alpha * (td_target - Q[state][action])

            state, action = next_state, next_action

            steps += 1

        all_rewards.append(total_reward)
        if reward == 100:  # goal 도달 시
            success_count += 1

        if (episode + 1) % log_interval == 0 or episode == episodes - 1:
            # V(s), π(s) 계산
            V = {s: max(Q[s].values()) for s in Q}
            policy = {s: max(Q[s], key=Q[s].get) for s in Q}
            plot_value_and_policy(V, policy, env.grid, episode, env.width, env.height, prefix='sarsa')

            # 로그 출력
            avg_reward = sum(all_rewards[-log_interval:]) / log_interval
            success_rate = success_count / log_interval * 100
            print(f"[Episode {episode+1}] Avg Reward: {avg_reward:.2f}, Success Rate: {success_rate:.1f}%")
            success_count = 0


    # 최종 policy return
    policy = {state: max(actions, key=actions.get) for state, actions in Q.items()}

    # plt.figure()
    # plt.plot(all_rewards)
    # plt.xlabel('Episode')
    # plt.ylabel('Total Reward')
    # plt.title('SARSA - Episode Rewards')
    # plt.grid(True)
    # plt.savefig('outputs/sarsa_rewards.png')
    # plt.close()

    return Q, policy
