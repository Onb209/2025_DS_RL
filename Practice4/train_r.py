import os
import argparse
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from env.gridworld_c2 import GridWorldEnv_c2
from algos.reinforce import REINFORCEAgent

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
    map_tensor = torch.tensor(map_img.transpose(2,0,1), dtype=torch.uint8)
    writer.add_image('Map', map_tensor, 0, dataformats='CHW')
    return map_img, H, W


def plot_continuous_policy(writer, env, agent, map_img, H, W, resolution, ep, step=3):
    """
    REINFORCE용 연속 정책 시각화: 배경 map 위에 subsample된 상태에서 평균 행동 방향으로 화살표 그리기
    """
    # 1) 그리드 지점 생성 (단위: 미터)
    xs = np.arange(resolution/2, W, resolution)
    ys = np.arange(resolution/2, H, resolution)
    grid_states = np.stack([[y, x] for y in ys for x in xs], axis=0)

    # 2) ray 정보를 최대 거리(감지 없음)로 패딩
    ray_len = env.ray_length_m
    ray_features = np.ones((grid_states.shape[0], env.ray_num), dtype=np.float32) * ray_len
    full_states = np.concatenate([grid_states, ray_features], axis=1)

    # 3) 정책 네트워크 호출
    st = torch.tensor(full_states, dtype=torch.float32, device=agent.device)
    with torch.no_grad():
        means, _ = agent.policy(st)
    means = means.cpu().numpy().reshape(len(ys), len(xs), 2)
    means = np.clip(means, -1.0, 1.0)

    # 4) 방향과 크기 계산
    norms = np.linalg.norm(means, axis=2)
    dirs = np.zeros_like(means)
    mask = norms > 1e-6
    dirs[mask] = means[mask] / norms[mask][..., None]

    # 5) subsampling
    X, Y = np.meshgrid(xs, ys)
    Xs = X[::step, ::step]
    Ys = Y[::step, ::step]
    Ds = dirs[::step, ::step]
    Ns = norms[::step, ::step]

    # 6) 시각화
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(map_img, origin='upper', extent=[0, W, H, 0], alpha=0.6, zorder=0)
    for (i, j), norm in np.ndenumerate(Ns):
        x = Xs[i, j]
        y = Ys[i, j]
        dy, dx = Ds[i, j]
        length = norm * 0.7
        ax.arrow(
            x, y,
            dx * length, dy * length,
            head_width=length * 0.7,
            head_length=length * 0.5,
            width=0.007
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
    # 1) SummaryWriter, 환경, 에이전트 초기화
    map_name = os.path.basename(args.map)
    if args.map in ['hw_map1.yaml', 'hw_map2.yaml', 'hw_map3.yaml']:
        num = map_name.split('_')[-1].split('.')[0]
        run_name = f'reinforce_hw_{num}'
    else:
        run_name = 'reinforce'
    writer = SummaryWriter(log_dir=os.path.join(args.logdir, run_name))

    config_path = os.path.join('configs', args.map)
    env = GridWorldEnv_c2(config_path)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = REINFORCEAgent(state_dim=state_dim, action_dim=action_dim)

    # 2) 맵 이미지 기록
    map_img, H, W = log_map(writer, env)

    # 3) 학습 루프
    for ep in range(1, args.episodes + 1):
        state = env.reset()
        agent.reset_episode()
        total_R = 0.0

        for t in range(args.max_steps):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=agent.device)
            action_tensor = agent.select_action(state_tensor)
            action = action_tensor.detach().cpu().numpy()

            next_state, reward, done, _ = env.step(action)
            total_R += reward

            # 버퍼에 보상 저장
            agent.rewards.append(reward)

            state = next_state
            if args.render:
                env.render()
            if done:
                break

        # 에피소드 종료 시 정책 업데이트 및 로깅
        loss = agent.finish_episode()
        writer.add_scalar('Loss', loss, ep)
        writer.add_scalar('Reward', total_R, ep)

        # 주기적 정책 시각화
        if ep % args.heatmap_interval == 0:
            plot_continuous_policy(writer, env, agent, map_img, H, W, args.resolution, ep)

        if ep % 100 == 0:
            print(f"[REINFORCEAgent] Episode: {ep}, Reward: {total_R:.2f}")

    # 4) 모델 저장 및 종료
    os.makedirs('checkpoints', exist_ok=True)
    agent.save(f'checkpoints/{run_name}.pth')
    writer.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--map', type=str, default='map1.yaml')
    p.add_argument('--episodes', type=int, default=1000)
    p.add_argument('--max-steps', type=int, default=100)
    p.add_argument('--render', action='store_true')
    p.add_argument('--logdir', type=str, default='runs')
    p.add_argument('--heatmap-interval', type=int, default=100)
    p.add_argument('--resolution', type=float, default=0.1)
    args = p.parse_args()
    train(args)


if __name__ == '__main__':
    main()
