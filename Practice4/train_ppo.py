import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from gymnasium import spaces

from env.gridworld_c2 import GridWorldEnv_c2

class GymWrapper(gym.Env):
    """
    Wrap GridWorldEnv_c2 to gym.Env API
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, config_path, headless=False):
        super(GymWrapper, self).__init__()
        self.env = GridWorldEnv_c2(config_path, headless=headless)
        # define spaces
        self.observation_space = spaces.Box(
            low=self.env.observation_space.low,
            high=self.env.observation_space.high,
            shape=self.env.observation_space.shape,
            dtype=self.env.observation_space.dtype
        )
        self.action_space = spaces.Box(
            low=self.env.action_space.low,
            high=self.env.action_space.high,
            shape=self.env.action_space.shape,
            dtype=self.env.action_space.dtype
        )

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        # gymnasium reset returns obs, info
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # gymnasium step returns obs, reward, terminated, truncated, info
        terminated = done
        truncated = False
        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        return self.env.render()

    def close(self):
        return self.env.close()

class RewardLogger(BaseCallback):
    """
    Callback for logging rewards per episode.
    """
    def __init__(self):
        super(RewardLogger, self).__init__()
        self.episode_rewards = []
        self._current_rewards = []

    def _on_step(self) -> bool:
        # stable-baselines3 stores 'rewards' or 'reward' in locals
        reward = self.locals.get('rewards', None)
        if reward is None:
            reward = self.locals.get('reward', 0)
        # append reward
        if isinstance(reward, list):
            self._current_rewards.append(reward[0])
        else:
            self._current_rewards.append(reward)
        # check done flag
        done = False
        dones = self.locals.get('dones', None)
        if dones is not None:
            done = dones[0]
        else:
            done = self.locals.get('terminated', False) or self.locals.get('truncated', False)
        if done:
            self.episode_rewards.append(sum(self._current_rewards))
            self._current_rewards = []
        return True


def make_env(map_file: str, headless: bool):
    def _init():
        return GymWrapper(os.path.join('configs', map_file), headless=headless)
    return _init


def main():
    parser = argparse.ArgumentParser(description='Test PPO on GridWorldEnv_c2')
    parser.add_argument('--map', type=str, default='map1.yaml', help='Map file under configs/')
    parser.add_argument('--timesteps', type=int, default=100_000, help='Total training timesteps')
    parser.add_argument('--headless', action='store_true', help='Headless mode (no rendering)')
    parser.add_argument('--logdir', type=str, default='ppo_runs', help='TensorBoard log directory')
    args = parser.parse_args()

    # create vectorized env
    env = DummyVecEnv([make_env(args.map, args.headless)])

    # callback to log rewards
    reward_logger = RewardLogger()

    # create PPO model
    model = PPO('MlpPolicy', env,
                verbose=1,
                tensorboard_log=args.logdir,
                ent_coef=0.01,
                learning_rate=3e-4,
                gamma=0.99)

    # train
    model.learn(total_timesteps=args.timesteps, callback=reward_logger)

    # save final model
    os.makedirs('ppo_checkpoints', exist_ok=True)
    model.save(f'ppo_checkpoints/ppo_{args.map[:-5]}.zip')

    # # plot episode rewards
    # plt.figure(figsize=(10,5))
    # rewards = reward_logger.episode_rewards
    # plt.plot(rewards, label='Episode Reward')
    # if len(rewards) >= 10:
    #     ma = np.convolve(rewards, np.ones(10)/10, mode='valid')
    #     plt.plot(range(9, len(rewards)), ma, label='10-Episode MA')
    # plt.xlabel('Episode')
    # plt.ylabel('Reward')
    # plt.title(f'PPO on GridWorldEnv_c2: {args.map}')
    # plt.legend()
    # plt.show()

if __name__ == '__main__':
    main()