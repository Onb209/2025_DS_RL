import os
import argparse
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

from env.gridworld_c2 import GridWorldEnv_c2

class GridWorldGymWrapper(gym.Env):
    """
    Gymnasium Env wrapper for GridWorldEnv_c2
    """
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, config_path: str, headless: bool = False):
        super().__init__()
        # 실제 env
        self.env = GridWorldEnv_c2(config_path, headless=headless)

        # 이미 underlying env에 정의된 space가 있으면 그대로 쓰기
        if hasattr(self.env, 'observation_space') and isinstance(self.env.observation_space, spaces.Space):
            self.observation_space = self.env.observation_space
        else:
            low  = np.array([0.0, 0.0], dtype=np.float32)
            high = np.array([self.env.height * self.env.cell_size_m,
                             self.env.width  * self.env.cell_size_m], dtype=np.float32)
            self.observation_space = spaces.Box(low, high, dtype=np.float32)

        if hasattr(self.env, 'action_space') and isinstance(self.env.action_space, spaces.Space):
            self.action_space = self.env.action_space
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def reset(self):
        obs = self.env.reset()
        return obs, {}  # Gymnasium API

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, False, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--map',        type=str, required=True,
                   help='config 파일 이름 (configs/*.yaml)')
    p.add_argument('--checkpoint', type=str, required=True,
                   help='불러올 PPO 체크포인트 (.zip)')
    p.add_argument('--episodes',   type=int, default=5,
                   help='렌더링할 에피소드 수')
    args = p.parse_args()

    cfg_path = os.path.join('configs', args.map)
    env = GridWorldGymWrapper(cfg_path, headless=False)

    # 1) 모델 로드 (env 인자 없이)
    model = PPO.load(args.checkpoint)
    # 2) wrapper 환경 연결
    model.set_env(env)

    for ep in range(1, args.episodes + 1):
        obs, _   = env.reset()
        done     = False
        total_r  = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, _, _ = env.step(action)
            total_r += r
            env.render()

        print(f"Episode {ep}: Total Reward = {total_r:.2f}")

    env.close()


if __name__ == '__main__':
    main()