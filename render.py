# render.py

import pickle
import time
import argparse
from env.gridworld_env import GridWorldEnv, Action

def load_policy(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def main(policy_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, required=True, help="Path to policy .pkl file")
    parser.add_argument('--width', type=int, default=6)
    parser.add_argument('--height', type=int, default=6)
    args = parser.parse_args()

    env = GridWorldEnv(width=args.width, height=args.height)
    policy = load_policy(policy_path)
    state = env.reset()

    while True:
        action = policy.get(tuple(state), None)
        if action is None:
            break
        state, _, done = env.step(action.value)
        env.render()
        time.sleep(0.25)
        if done:
            time.sleep(1.0)
            state = env.reset()

if __name__ == "__main__":
    main()
