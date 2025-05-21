import random
from collections import defaultdict
from env.gridworld_env import Action
from tqdm import tqdm  
import time
from algos.dynamic_programming import plot_value_and_policy

# first_visit = True, use_incremental_mean = False
# → First-Visit MC
#   - 한 에피소드 내에서 (state, action) 쌍이 처음 나타난 경우에만 업데이트
#   - returns[(s,a)] 리스트에 모든 G를 저장 후 평균으로 Q(s,a) 계산
#
# first_visit = False, use_incremental_mean = False
# → Every-Visit MC
#   - (state, action) 쌍이 에피소드에서 여러 번 나와도 모두 업데이트
#   - G 값들을 리스트에 저장 후 평균으로 Q(s,a) 계산
#
# first_visit = True, use_incremental_mean = True
# → First-Visit MC + Incremental Mean
#   - 한 에피소드 내 처음 등장한 (state, action)에만 업데이트
#   - G 값 평균을 저장하지 않고, 점진적 방식으로 Q(s,a) 갱신
#     Q(s,a) ← Q(s,a) + (1 / N(s,a)) * (G - Q(s,a))
#
# first_visit = False, use_incremental_mean = True
# → Every-Visit MC + Incremental Mean
#   - (state, action) 쌍이 나타날 때마다 모두 업데이트
#   - G를 즉시 incremental 방식으로 반영하여 Q(s,a) 갱신


def monte_carlo(env, episodes=5000, gamma=0.99, epsilon=0.1, render=False, log_interval=100, 
                first_visit=True, use_incremental_mean=False):
    Q = defaultdict(lambda: {a: 0.0 for a in Action}) # Q(s,a): state-action value function
    returns = defaultdict(list) 
    counts = defaultdict(int)  # For incremental mean. visit 횟수 N(s,a)

    all_rewards = []
    success_count = 0

    for episode in tqdm(range(episodes), desc="Training Monte Carlo"):
        episode_data = []
        state = tuple(env.reset())
        done = False
        total_reward = 0

        # 에피소드 생성
        while not done:
            if render and episode % log_interval == 0:
                env.render()
                time.sleep(0.05)

            # epsilon-greedy policy
            if random.random() < epsilon:
                action = random.choice(list(Action)) # exploration
            else:
                action = max(Q[state], key=Q[state].get) # exploitation

            next_state, reward, done = env.step(action.value)
            episode_data.append((state, action, reward)) # 한 에피소드는 (s, a, r)의 시퀀스로 저장
            total_reward += reward
            state = tuple(next_state)

        all_rewards.append(total_reward)
        if reward == 100:
            success_count += 1

        G = 0
        visited = set() # first-visit 용
        
        for t in reversed(range(len(episode_data))):
            state, action, reward = episode_data[t]
            G = gamma * G + reward

            if not first_visit or (state, action) not in visited: # Every-visit 또는 First-visit에서 처음 등장했을 때
                
                if use_incremental_mean: # 기존 Q값과 G의 차이를 학습률 α로 보정하여 업데이트
                    counts[(state, action)] += 1
                    alpha = 1 / counts[(state, action)]
                    Q[state][action] += alpha * (G - Q[state][action])
                
                else: # 모든 G 값을 저장한 후, 평균을 구해 Q를 업데이트
                    returns[(state, action)].append(G)
                    Q[state][action] = sum(returns[(state, action)]) / len(returns[(state, action)])
                
                visited.add((state, action))

        # 로그 출력 및 시각화
        if (episode + 1) % log_interval == 0:
            avg_reward = sum(all_rewards[-log_interval:]) / log_interval
            success_rate = success_count / log_interval * 100
            print(f"[Episode {episode+1}] Avg Reward: {avg_reward:.2f}, Success Rate: {success_rate:.1f}%")
            success_count = 0

            # Q → V, π 변환 및 시각화
            V = {s: max(a_vals.values()) for s, a_vals in Q.items()}
            policy = {s: max(a_vals, key=a_vals.get) for s, a_vals in Q.items()}
            plot_value_and_policy(V, policy, env.grid, episode + 1, env.width, env.height, prefix="mc")


    policy = {state: max(actions, key=actions.get) for state, actions in Q.items()}
    return Q, policy
