import argparse
import pickle
from env.gridworld_env import GridWorldEnv
from env.gridworld_mdp import GridWorldMDP

from algos.dynamic_programming import policy_iteration, value_iteration

def save_policy(pi, filename):
    with open(filename, 'wb') as f:
        pickle.dump(pi, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, required=True,
                        choices=["vi", "pi"],
                        help="Choose algorithm: vi, pi")
    parser.add_argument('--width', type=int, default=6)
    parser.add_argument('--height', type=int, default=6)
    parser.add_argument('--render', action='store_true', help="Render environment during training")
    args = parser.parse_args()

    env = GridWorldEnv(width=args.width, height=args.height)

    print(f"=== Running {args.algo.upper()} ===")
    
    mdp = GridWorldMDP(env)
    if args.algo == 'vi':
        _, pi = value_iteration(mdp)
    else:
        _, pi = policy_iteration(mdp)

    save_path = f"checkpoints/policy_{args.algo}.pkl"
    save_policy(pi, save_path)
    print(f"Policy saved to {save_path}")

if __name__ == "__main__":
    main()
