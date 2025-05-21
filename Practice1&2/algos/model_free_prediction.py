import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
from env.gridworld_env import Action, TileType

def random_policy(state):
    return random.choice(list(Action))

# Monte Carlo 방법은 에피소드가 끝난 후 전체 보상 합(G)을 계산하여 V(s)를 추정
# First-visit
def monte_carlo_prediction(env, policy, episodes=1000, gamma=0.99):
    V = defaultdict(float)
    returns = defaultdict(list)

    for _ in range(episodes):
        state = tuple(env.reset())
        episode = []

        done = False

        # 한 에피소드 내에서 (state, reward) 튜플을 저장
        while not done:
            action = policy(state)
            next_state, reward, done = env.step(action.value)
            episode.append((state, reward))
            state = tuple(next_state)

        G = 0
        visited = set()
        # 에피소드 reward를 뒤에서부터 역순으로 계산 (누적 reward 계산 효율적으로 하기 위함)
        for t in reversed(range(len(episode))):
            s_t, r_t = episode[t]
            G = gamma * G + r_t # 전체 retrun G 계산
            if s_t not in visited: # 한 에피소드 내에서 처음 등장하는 시점의 return만 사용
                returns[s_t].append(G)
                V[s_t] = sum(returns[s_t]) / len(returns[s_t]) # state별 return의 평균
                visited.add(s_t)

    return V

# 한 스텝씩 환경에서 받은 보상과 다음 상태의 가치 추정치로 V(s)를 점진적으로 업데이트
def td0_prediction(env, policy, episodes=1000, alpha=0.1, gamma=0.99):
    V = defaultdict(float)

    for _ in range(episodes):
        state = tuple(env.reset())
        done = False

        while not done: # 한 step마다 V(s)를 점진적으로 업데이트
            action = policy(state)
            next_state, reward, done = env.step(action.value)
            next_state = tuple(next_state)

            # TD(0) update
            td_target = reward + gamma * V[next_state]
            V[state] += alpha * (td_target - V[state])

            state = next_state

    return V

# Monte Carlo vs TD(0): 여러 번 반복 학습하여 각 상태에 대한 평균 값 추정과 분산 비교
# 각 알고리즘을 runs만큼 반복 실행 → 결과 통계 분석
def run_prediction_experiment(env, episodes=1000, runs=30, gamma=0.99, alpha=0.1):
    all_mc_values = []
    all_td_values = []

    # runs번 실행 → Monte Carlo와 TD(0)의 여러 추정값을 수집
    for _ in range(runs):
        V_mc = monte_carlo_prediction(env, random_policy, episodes, gamma)
        V_td = td0_prediction(env, random_policy, episodes, alpha, gamma)
        all_mc_values.append(V_mc)
        all_td_values.append(V_td)

    all_states = set().union(*[v.keys() for v in all_mc_values + all_td_values])
    all_states = sorted(all_states)

    mc_means, td_means = [], []
    mc_vars, td_vars = [], []

    # 각 상태에 대해 Monte Carlo 또는 TD가 추정한 V(s)의 평균과 분산을 계산
    for s in all_states:
        mc_vals = [v.get(s, 0.0) for v in all_mc_values]
        td_vals = [v.get(s, 0.0) for v in all_td_values]

        mc_means.append(np.mean(mc_vals))
        td_means.append(np.mean(td_vals))
        mc_vars.append(np.var(mc_vals))
        td_vars.append(np.var(td_vals))

    # 그래프 그리기
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    x_labels = [str(s) for s in all_states]

    axs[0].plot(x_labels, mc_means, label="Monte Carlo", marker='o')
    axs[0].plot(x_labels, td_means, label="TD(0)", marker='x')
    axs[0].set_ylabel("Mean V(s)")
    axs[0].set_title("Monte Carlo vs TD(0) Value Estimates (Mean)")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(x_labels, mc_vars, label="Monte Carlo", marker='o')
    axs[1].plot(x_labels, td_vars, label="TD(0)", marker='x')
    axs[1].set_ylabel("Variance")
    axs[1].set_xlabel("State")
    axs[1].set_title("Monte Carlo vs TD(0) Value Estimates (Variance)")
    axs[1].legend()
    axs[1].grid(True)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("outputs/bias_variance_comparison.png")
    plt.close()

