import argparse
import pickle
from env.gridworld_env import GridWorldEnv
from env.gridworld_mdp import GridWorldMDP

# 알고리즘 로드
from algos.dynamic_programming import policy_iteration, value_iteration
from algos.monte_carlo import monte_carlo
# from algos.td0 import td0
from algos.sarsa import sarsa
from algos.q_learning import q_learning

# policy를 pkl 파일로 저장
def save_policy(pi, filename):
    with open(filename, 'wb') as f:
        pickle.dump(pi, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, required=True,
                        choices=["vi", "pi", "mc", "td0", "sarsa", "q_learning"],
                        help="Choose algorithm: vi, pi")
    parser.add_argument('--size', type=int, default=5)
    parser.add_argument('--render', action='store_true', help="Render environment during training")
    args = parser.parse_args()

    map_name = f"map_{args.size}.json"
    env = GridWorldEnv(width=args.size, height=args.size, map_file=map_name)

    print(f"=== Running {args.algo.upper()} ===")
    
    if args.algo in ['vi', 'pi']:
        mdp = GridWorldMDP(env)
        if args.algo == 'vi':
            _, pi = value_iteration(mdp)
        else:
            _, pi = policy_iteration(mdp)
    else:
        # model-free methods
        if args.algo == 'mc':
            _, pi = monte_carlo(env, render=args.render)
        # elif args.algo == 'td0':
        #     td0(env, render=args.render)
        #     return
        elif args.algo == 'sarsa':
            _, pi = sarsa(env, render=args.render)
        elif args.algo == 'q_learning':
            _, pi = q_learning(env, render=args.render)

    save_path = f"checkpoints/policy_{args.algo}.pkl"
    save_policy(pi, save_path)
    print(f"Policy saved to {save_path}")

if __name__ == "__main__":
    main() 



