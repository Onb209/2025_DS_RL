import os
import argparse
import importlib
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from env import gridworld_c1, gridworld_c2

# 알고리즘 이름 ↔ 클래스 이름 매핑
CLASS_MAP = {
    'deepsarsa': 'DeepSARSAAgent',
    'dqn':       'DQNAgent',
    'reinforce': 'REINFORCEAgent',
}

def log_map(writer, env):
    """
    환경의 grid 정보를 바탕으로 배경 맵 이미지를 TensorBoard에 기록.
    """
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
        map_img[grid == v] = c
    # CHW, uint8
    map_tensor = torch.tensor(map_img.transpose(2,0,1), dtype=torch.uint8)
    writer.add_image('Map', map_tensor, 0, dataformats='CHW')
    return map_img, H, W

def compute_value_grid(env, qnet, state_scale, device='cpu'):
    """
    모든 상태 좌표에서 V(s) 계산, 격자 좌표 반환.
    """
    H, W = env.height, env.width

    # 1) 모든 셀 좌표 생성
    states = np.stack([[r, c] for r in range(H) for c in range(W)], axis=0).astype(np.float32)

    # 2) 에이전트와 동일한 정규화 적용
    states_norm = states / state_scale  # shape (H*W, 2)

    qnet.eval()  # 혹시 batchnorm/dropout이 있다면 평가 모드로
    with torch.no_grad():
        st = torch.tensor(states_norm, dtype=torch.float32, device=device)
        q = qnet(st).cpu().numpy()  # shape (H*W, A)

    # 3) 그리드 형태로 되돌리기
    q = q.reshape(H, W, -1)
    values = q.max(axis=2)

    return values, q

def plot_value_heatmap(writer, values, xs, ys, ep):
    """
    StateValueHeatmap Figure 생성 및 TensorBoard 기록.
    """
    fig, ax = plt.subplots(figsize=(4,4))
    extent = [
        xs[0] - (xs[1]-xs[0])/2,
        xs[-1]+ (xs[1]-xs[0])/2,
        ys[-1]+ (ys[1]-ys[0])/2,
        ys[0] - (ys[1]-ys[0])/2
    ]
    im = ax.imshow(values,
                   origin='upper',
                   interpolation='bilinear',
                   cmap='viridis',
                   extent=extent)
    fig.colorbar(im, ax=ax, label='V(s)')
    ax.set_title('StateValueHeatmap')
    writer.add_figure('StateValueHeatmap', fig, global_step=ep)
    plt.close(fig)

def plot_discrete_policy(writer, env, qvals, xs, ys, map_img, H, W, ep, step=3):
    """
    DeepSARSA/DQN 용: 각 격자에서 argmax Q 행동 방향을 화살표로 표시.
    """
    # 화살표 방향 (unit vectors over 8 actions)
    deltas = env.deltas  # (8,2) in meters
    norms = np.linalg.norm(deltas, axis=1, keepdims=True)
    unit = deltas / norms * 0.2
    # 최적 행동 인덱스
    actions = qvals.argmax(axis=2)  # (n_y, n_x)
    # 벡터 컴포넌트
    U = unit[actions][:,:,1]
    V = -unit[actions][:,:,0]
    # subsample
    X, Y = np.meshgrid(xs, ys)
    Xs = X[::step, ::step]; Ys = Y[::step, ::step]
    Us = U[::step, ::step]; Vs = V[::step, ::step]
    # plot
    fig, ax = plt.subplots(figsize=(8, 8))
    # map background
    ax.imshow(map_img, origin='upper', extent=[0,W,H,0], alpha=0.6, zorder=0)
    ax.quiver(Xs, Ys, Us, Vs,
              color='blue', scale_units='xy', scale=1,
              width=0.005, zorder=1)
    ax.set_xlim(0,W); ax.set_ylim(H,0)
    ax.set_title('PolicyArrows (Discrete)')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    writer.add_figure('PolicyArrows', fig, global_step=ep)
    plt.close(fig)

def plot_continuous_policy(writer, env, agent, map_img, H, W, resolution, ep, step=3):
    """
    REINFORCE용 연속 정책 시각화:
    - map 배경 위에, subsample된 상태에서
      * 평균 행동 방향으로 ax.arrow 그리기
      * 화살표 길이 ∝ action norm (scaled by 0.1)
      * shaft width는 0.01로 고정
    """
    # 1) 샘플링 격자 생성
    xs = np.arange(resolution/2, W, resolution)
    ys = np.arange(resolution/2, H, resolution)
    grid_states = np.stack([[y, x] for y in ys for x in xs], axis=0)

    # 2) 정책 네트워크에서 mean value 추출
    st = torch.tensor(grid_states, dtype=torch.float32, device=agent.device)
    with torch.no_grad():
        means, _ = agent.policy(st)               # (N, 2)
    means = means.cpu().numpy().reshape(len(ys), len(xs), 2)
    means = np.clip(means, -1.0, 1.0)

    # 3) norm 및 방향(unit) 계산
    norms = np.linalg.norm(means, axis=2)        # (n_y, n_x)
    dirs = np.zeros_like(means)
    mask = norms > 1e-6
    dirs[mask] = means[mask] / norms[mask][..., None]

    # 4) subsample
    X, Y = np.meshgrid(xs, ys)
    Xs = X[::step, ::step]
    Ys = Y[::step, ::step]
    Ds = dirs[::step, ::step]                    # (m, n, 2)
    Ns = norms[::step, ::step]                   # (m, n)

    # 5) 그리기
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(map_img, origin='upper', extent=[0, W, H, 0], alpha=0.6, zorder=0)

    # 각 화살표마다
    for (i, j), norm in np.ndenumerate(Ns):
        x = Xs[i, j]
        y = Ys[i, j]
        dy, dx = Ds[i, j]  # unit vector [dr, dc]
        length = norm * 0.1  # 0~1 norm에 0.1 곱해 스케일
        # dx, dy 방향으로 화살표
        ax.arrow(
            x, y,
            dx * length, dy * length,   # 화면 y축 반전 고려
            head_width=length * 0.7,
            head_length=length * 0.5,
            fc='blue', ec='blue',
            width=0.007                  # shaft width 고정
        )

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect('equal')
    ax.set_title('PolicyArrows (Continuous)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    writer.add_figure('PolicyArrows', fig, global_step=ep)
    plt.close(fig)




def train(args):
    # 1) writer, env, agent 초기화
    writer = SummaryWriter(log_dir=os.path.join(args.logdir, args.algo))
    mod = importlib.import_module(f'algos.{args.algo}')
    AgentClass = getattr(mod, CLASS_MAP[args.algo])
    config_path = os.path.join('configs', args.map)
    if args.algo == 'reinforce':
        env = gridworld_c2.GridWorldEnv_c2(config_path)
    else:
        env = gridworld_c1.GridWorldEnv_c1(config_path)
    agent = AgentClass(env)

    # 2) map image 기록
    map_img, H, W = log_map(writer, env)

    # 3) 학습 루프
    for ep in range(1, args.episodes+1):
        state = env.reset()
        agent.reset_episode()
        total_R = 0.0

        # ε
        if hasattr(agent,'epsilon'):
            writer.add_scalar('Epsilon', agent.epsilon, ep)

        # episode rollout
        action = agent.select_action(state)
        for t in range(args.max_steps):
            next_s, r, done, _ = env.step(action)
            total_R += r
            if args.algo == 'deepsarsa':
                next_a = agent.select_action(next_s)
                loss = agent.learn(state, action, r / 100., next_s, next_a, done)
            else:
                loss = agent.learn(state, action, r, next_s, done)
                next_a = agent.select_action(next_s)
            if loss is not None:
                writer.add_scalar('Loss', loss, ep)
            state, action = next_s, next_a
            if args.render: env.render(tick=5000)
            if done: break

        # finish
        if args.algo == 'reinforce':
            loss = agent.finish_episode()
            writer.add_scalar('Loss', loss, ep)
        else:
            agent.finish_episode(ep)

        writer.add_scalar('Reward', total_R, ep)

        # periodic visualization
        if ep % args.heatmap_interval==0:
            if args.algo in ['deepsarsa','dqn']:
                values, xs, ys, qvals = compute_value_grid(
                    env, agent.qnet, args.resolution, device=agent.device)
                plot_value_heatmap(writer, values, xs, ys, ep)
                plot_discrete_policy(writer, env, qvals, xs, ys, map_img, H, W, ep)
            else:  # reinforce
                plot_continuous_policy(writer, env, agent,
                                       map_img, H, W,
                                       args.resolution, ep)

        if ep % 100 == 0:
            disp = f", Epsilon: {agent.epsilon:.3f}" if hasattr(agent,'epsilon') else ''
            print(f"[{CLASS_MAP[args.algo]}] Episode: {ep}, Reward: {total_R:.2f}{disp}")

    # 4) save & close
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
    p.add_argument('--resolution', type=float, default=0.1)
    args = p.parse_args()
    train(args)

if __name__=='__main__':
    main()
