# render.py

import pickle
import time
import argparse
import random
from env.gridworld_env import GridWorldEnv, Action

def load_policy(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, required=False, help="Path to policy .pkl file")
    parser.add_argument('--size', type=int, default=6)
    parser.add_argument('--random', action='store_true', help="Use randomly generated map")
    parser.add_argument('--map', type=str, default=None,
                        help="Path to map JSON file (optional, overrides --size)")
    args = parser.parse_args()

    map_name = None
    if not args.random:
        map_name = args.map if args.map else f"map_{args.size}.json"

    env = GridWorldEnv(width=args.size, height=args.size, map_file=map_name)
    
    policy = None
    if args.policy:
        policy = load_policy(args.policy)
    
    state = env.reset()

    while True:
        if policy:
            action = policy.get(tuple(state), None)
            if action is None:
                break
        else:
            action = random.choice(list(Action))
        state, _, done = env.step(action.value)
        env.render()
        time.sleep(0.25)
        if done:
            time.sleep(1.0)
            state = env.reset()

if __name__ == "__main__":
    main()
