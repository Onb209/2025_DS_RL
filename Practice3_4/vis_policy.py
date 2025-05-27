import os
import sys
import argparse
import importlib
import yaml
import numpy as np
import torch
import matplotlib
# 헤드리스 환경에서 실행할 때 'Agg' 백엔드 사용
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 프로젝트 루트를 경로에 추가하여 모듈 임포트 가능하도록 설정
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)


def load_environment(agent_type: str, config_path: str):
    """
    에이전트 종류에 따라 올바른 환경 클래스를 임포트하고 반환합니다.
    """
    if agent_type == 'reinforce':
        env_mod = importlib.import_module('env.gridworld_c2')
        EnvClass = getattr(env_mod, 'GridWorldEnv_c2')
    else:
        env_mod = importlib.import_module('env.gridworld_c1')
        EnvClass = getattr(env_mod, 'GridWorldEnv_c1')
    return EnvClass(config_path)


def load_q_network(agent_type: str, checkpoint_path: str, device='cpu'):
    """
    에이전트 종류에 맞는 QNetwork 클래스를 동적으로 임포트하여 체크포인트 로드 후 반환.
    """
    mod = importlib.import_module(f'algos.{agent_type}')
    QNetClass = getattr(mod, 'QNetwork', None)
    if QNetClass is None:
        raise ImportError(f"algos.{agent_type}에 QNetwork 클래스가 없습니다.")
    qnet = QNetClass().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        qnet.load_state_dict(ckpt['model_state_dict'])
    else:
        qnet.load_state_dict(ckpt)
    qnet.eval()
    return qnet


def compute_value_action_grid(env, qnet: torch.nn.Module, resolution: float, device='cpu'):
    """
    해상도(resolution)만큼 격자 샘플을 생성하여
    V(s)=max_a Q(s,a) 및 최적 행동 argmax_a Q(s,a)를 계산.
    Returns:
        values: np.ndarray shape (n_y, n_x)
        actions: np.ndarray shape (n_y, n_x)
        xs, ys: 1D arrays of x, y 좌표
    """
    H, W = env.height, env.width
    xs = np.arange(resolution/2, W, resolution)
    ys = np.arange(resolution/2, H, resolution)
    grid_states = np.stack([[y, x] for y in ys for x in xs], axis=0)
    states_tensor = torch.tensor(grid_states, dtype=torch.float32, device=device)
    with torch.no_grad():
        qvals = qnet(states_tensor)  # (N, A)
    qvals = qvals.cpu().numpy().reshape(len(ys), len(xs), -1)
    values = qvals.max(axis=2)
    actions = qvals.argmax(axis=2)
    return values, actions, xs, ys


def plot_value_and_policy(values: np.ndarray,
                          actions: np.ndarray,
                          xs: np.ndarray,
                          ys: np.ndarray,
                          env, agent: str,
                          cmap: str = 'hot', vmin=None, vmax=None,
                          output_dir: str = './outputs'):
    """
    히트맵 위에 최적 행동을 화살표로 표현합니다.
    delta의 y축만 반전하여 올바른 방향을 그립니다.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    # 히트맵
    im = ax.imshow(values,
                   origin='upper',
                   interpolation='bilinear',
                   cmap=cmap,
                   vmin=vmin, vmax=vmax,
                   extent=[xs[0] - (xs[1]-xs[0])/2,
                           xs[-1] + (xs[1]-xs[0])/2,
                           ys[-1] + (ys[1]-ys[0])/2,
                           ys[0] - (ys[1]-ys[0])/2])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('V(s)')

    # 화살표(정책): delta의 y축만 반전
    deltas = env.deltas  # (8,2) shape: [dr, dc]
    norms = np.linalg.norm(deltas, axis=1, keepdims=True)
    unit_deltas = deltas / norms * 0.15
    # 메쉬 그리드
    X, Y = np.meshgrid(xs, ys)
    # quiver 방향 성분: x축은 dc, y축은 -dr (y 반전)
    U = unit_deltas[actions][:,:,1]
    V = -unit_deltas[actions][:,:,0]
    ax.quiver(X, Y, U, V, color='cyan', scale_units='xy', scale=0.5, width=0.005)

    ax.set_title(f"Value & Policy ({agent})")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    out_path = os.path.join(output_dir, f'{agent}_value_policy.png')
    plt.savefig(out_path, bbox_inches='tight')
    print(f"Value & Policy plot saved: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='훈련된 Q-network로부터 상태-가치 및 정책 시각화')
    parser.add_argument('--agent', choices=['deepsarsa', 'dqn'], required=True)
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='checkpoints/xxx.pth')
    parser.add_argument('--map', type=str, default='map1.yaml')
    parser.add_argument('--resolution', type=float, default=0.5,
                        help='샘플링 해상도 (m)')
    parser.add_argument('--cmap', type=str, default='hot')
    parser.add_argument('--vmin', type=float, default=None)
    parser.add_argument('--vmax', type=float, default=None)
    parser.add_argument('--output', type=str, default='./outputs')
    args = parser.parse_args()

    map_path = os.path.join('configs', args.map)
    ckpt_path = os.path.join('checkpoints', args.checkpoint)

    env = load_environment(args.agent, map_path)
    qnet = load_q_network(args.agent, ckpt_path, device='cpu')

    values, actions, xs, ys = compute_value_action_grid(env, qnet,
                                                         resolution=args.resolution, device='cpu')
    plot_value_and_policy(values, actions, xs, ys,
                          env, args.agent,
                          cmap=args.cmap, vmin=args.vmin, vmax=args.vmax,
                          output_dir=args.output)

if __name__ == '__main__':
    main()
