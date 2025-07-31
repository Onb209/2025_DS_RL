from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from custom_walker2d import CustomEnvWrapper
from stable_baselines3.common.callbacks import CheckpointCallback

# TODO: Modify if necessary
# Adjust based on available CPU cores (Windows CMD: wmic cpu get NumberOfLogicalProcessors)
N_ENVS = 4

def make_env(bump_practice = False, bump_challenge=False):
    def _init():
        env = CustomEnvWrapper(render_mode=None, bump_practice=bump_practice, bump_challenge=bump_challenge)
        return env
    return _init

# TODO: Modify if necessary
policy_kwargs = dict(
    net_arch=[dict(pi=[128, 64, 64], vf=[128, 64, 64])],
    log_std_init=-1.0 
)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--bump_practice", action="store_true", help="Enable bumping") # For bump practice
parser.add_argument("--bump_challenge", action="store_true", help="Enable bumping") # For bump challenge
args = parser.parse_args()

if __name__ == "__main__":
    num_cpu = N_ENVS
    env = SubprocVecEnv([make_env(bump_practice=args.bump_practice, bump_challenge=args.bump_challenge) for _ in range(num_cpu)])
    env = VecMonitor(env)

    if args.bump_practice:
        folder_name = "bump_practice"
    elif args.bump_challenge:
        folder_name = "bump_challenge"
    else:
        folder_name = "walker_model"

    save_path = f'./checkpoints/{folder_name}/'

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=save_path,
        name_prefix="walker_model"
    )
    
    # TODO: Modify if necessary
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/", policy_kwargs=policy_kwargs, device="cpu", learning_rate=0.0001)
    
    model.learn(total_timesteps=10000000000, callback=checkpoint_callback)
    model.save("ppo_custom_walker2d_parallel")