import os
import argparse
import importlib
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from env import gridworld_c2, gridworld_d2

# 알고리즘 이름 ↔ 클래스 이름 매핑
CLASS_MAP = {
    'deepsarsa': 'DeepSARSAAgent',
    'dqn':       'DQNAgent',
    'reinforce': 'REINFORCEAgent',
}


def log_map(writer, env):
    grid = env.grid
    H, W = grid.shape
    color_map = {
        0: [220,220,220],  # normal
        1: [50,50,50],     # wall
        2: [200,0,0],      # trap
        3: [0,200,0],      # goal
    }
    map_img = np.zeros((H, W, 3), dtype=np.uint8)
    for v, c in color_map.items():
        map_img[grid==v] = c
    map_tensor = torch.tensor(map_img.transpose(2,0,1), dtype=torch.uint8)
    writer.add_image('Map', map_tensor, 0, dataformats='CHW')
    return map_img, H, W


def compute_value_grid_discrete(env, qnet, algo, device='cpu',):
    H, W = env.height, env.width
    # 1) 모든 셀 좌표 생성
    states = np.stack([[r, c] for r in range(H) for c in range(W)], axis=0).astype(np.float32)
    # 2) [0,1] 범위로 정규화
    if algo == 'deepsarsa':
        states_norm = states / np.array([H, W], dtype=np.float32)
    else:
        states_norm = states

    qnet.eval()
    with torch.no_grad():
        st = torch.tensor(states_norm, dtype=torch.float32, device=device)
        q = qnet(st).cpu().numpy()   # (H*W, A)
    # 3) 그리드 형태로 복원
    q = q.reshape(H, W, -1)
    values = q.mean(axis=2)
    return values, q



def plot_discrete_policy(writer, env, qvals, map_img, H, W, ep, step=1):
    """
    원래 방식 그대로, 상태 정규화 없이
    - action_dirs에서 dr,dc를 읽어 와
    - unit 벡터(길이 20% cell)로 화살표 표시
    """
    # 1) 8방향 단위 벡터 계산 (dr, dc)
    dirs = np.array(env.action_dirs)            # shape (8,2)
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    unit = dirs / norms * 0.4                    # 화살표 길이: 셀 크기의 20%

    # 2) 각 셀마다 argmax Q 행동 인덱스
    actions = qvals.argmax(axis=2)               # shape (H, W)

    # 3) U,V 성분 뽑기
    U = unit[actions][:,:,1]                     # x축(열) 방향
    V = -unit[actions][:,:,0]                    # y축(행) 방향 (origin='upper' 고려)

    # 4) 격자 중심 좌표 생성
    xs = np.arange(0.5, W, 1)
    ys = np.arange(0.5, H, 1)
    X, Y = np.meshgrid(xs, ys)
    Xs = X[::step, ::step]
    Ys = Y[::step, ::step]
    Us = U[::step, ::step]
    Vs = V[::step, ::step]

    # 5) 그리기
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(map_img, origin='upper', extent=[0, W, H, 0], alpha=0.6)
    ax.quiver(
        Xs, Ys,
        Us, Vs,
        color='blue', scale=1, scale_units='xy', width=0.005
    )
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_title('PolicyArrows (Discrete)')
    writer.add_figure('PolicyArrows', fig, global_step=ep)
    plt.close(fig)

def train(args):
    writer = SummaryWriter(log_dir=os.path.join(args.logdir, args.algo))
    mod = importlib.import_module(f'algos.{args.algo}')
    AgentClass = getattr(mod, CLASS_MAP[args.algo])
    config_path = os.path.join('configs', args.map)

    # 환경 선택
    if args.algo in ['deepsarsa', 'dqn']:
        env = gridworld_d2.GridWorldEnvDiscrete(config_path)
    else:
        env = gridworld_c2.GridWorldEnv_c2(config_path)

    agent = AgentClass(env)
    map_img, H, W = log_map(writer, env)

    for ep in range(1, args.episodes+1):
        state = env.reset()
        agent.reset_episode()
        total_R = 0.0

        if hasattr(agent, 'epsilon'):
            writer.add_scalar('Epsilon', agent.epsilon, ep)

        if args.algo == 'reinforce':
            action = agent.select_action(state)
            for t in range(args.max_steps):
                next_s, r, done, _ = env.step(action)
                total_R += r
                loss = agent.learn(state, action, r, next_s, None, done)
                state = next_s
                action = agent.select_action(state)
                if done: break
            loss = agent.finish_episode()
            writer.add_scalar('Loss', loss, ep)
        else:
            action = agent.select_action(state)
            for t in range(args.max_steps):
                next_s, r, done, _ = env.step(action)
                total_R += r
                if args.algo == 'deepsarsa':
                    next_a = agent.select_action(next_s)
                    loss = agent.learn(state, action, r / 100, next_s, next_a, done)
                else:
                    loss = agent.learn(state, action, r / 100, next_s, done)
                    next_a = agent.select_action(next_s)
                state, action = next_s, next_a
                if loss is not None:
                    writer.add_scalar('Loss', loss, ep)
                if done: break
            agent.finish_episode(ep)

        writer.add_scalar('Reward', total_R, ep)

        # 시각화
        if ep % args.heatmap_interval == 0:
            if args.algo in ['deepsarsa', 'dqn']:
                values, qvals = compute_value_grid_discrete(env, agent.qnet, args.algo, device=agent.device)
                # heatmap
                fig, ax = plt.subplots(figsize=(4,4))
                im = ax.imshow(values, origin='upper', cmap='viridis')
                writer.add_figure('StateValueHeatmap', fig, global_step=ep)
                plt.close(fig)
                plot_discrete_policy(writer, env, qvals, map_img, H, W, ep)
            else:
                # continuous policy visualization unchanged
                from train import plot_continuous_policy
                plot_continuous_policy(writer, env, agent, map_img, H, W, args.resolution, ep)

        if ep % 100 == 0:
            disp = f", Epsilon: {agent.epsilon:.3f}" if hasattr(agent, 'epsilon') else ''
            print(f"[{args.algo}] Episode: {ep}, Reward: {total_R:.2f}" + disp)

    os.makedirs('checkpoints', exist_ok=True)
    agent.save(f'checkpoints/{args.algo}.pth')
    writer.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--algo', choices=list(CLASS_MAP.keys()), required=True)
    p.add_argument('--map', type=str, default='map1.yaml')
    p.add_argument('--episodes', type=int, default=1000)
    p.add_argument('--max-steps', type=int, default=100)
    p.add_argument('--render', action='store_true')
    p.add_argument('--logdir', type=str, default='runs')
    p.add_argument('--heatmap-interval', type=int, default=100)
    p.add_argument('--resolution', type=float, default=1.0)
    args = p.parse_args()
    train(args)

if __name__ == '__main__':
    main()
