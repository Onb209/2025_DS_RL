import argparse
import pickle
import time
from env.gridworld_env import GridWorldEnv, Action
from env.gridworld_mdp import GridWorldMDP

from algos.dynamic_programming import policy_iteration, value_iteration
from algos.monte_carlo import monte_carlo
from algos.model_free_prediction import run_prediction_experiment
from algos.sarsa import sarsa
from algos.q_learning import q_learning

def evaluate_single_episode(env, policy, render=False, max_steps=100):
    state = env.reset()
    done = False
    steps = 0

    while not done and steps < max_steps:
        action = policy.get(tuple(state), None)
        if action is None:
            if render:
                print(f"No action found for state {state}. Terminating episode.")
            break
        state, reward, done = env.step(action.value)
        steps += 1
        if render:
            env.render()
            time.sleep(0.25)

    if render:
        time.sleep(1.0)

    if steps >= max_steps:
        if render:
            print(f"Episode terminated due to step limit ({max_steps}).")

    return reward == 100  # 성공 여부


def train_policy(env, algo, render=False):
    if algo in ['vi', 'pi']:
        mdp = GridWorldMDP(env)
        return value_iteration(mdp)[1] if algo == 'vi' else policy_iteration(mdp)[1]
    if algo == 'mf_pred':
        run_prediction_experiment(env)
        return None
    if algo == 'mc':
        return monte_carlo(env)[1]
    if algo == 'sarsa':
        return sarsa(env, render=render)[1]
    if algo == 'q_learning':
        return q_learning(env, render=render)[1]
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, required=True,
                        choices=["vi", "pi", "mf_pred", "mc", "sarsa", "q_learning"])
    parser.add_argument('--size', type=int, default=6)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--map', type=str, default=None)
    parser.add_argument('--train_runs', type=int, default=10)
    args = parser.parse_args()

    map_name = args.map if args.map else f"map_{args.size}.json"
    print(f"=== Running {args.train_runs} training + single-episode evaluation cycles with {args.algo.upper()} ===")

    success_count = 0

    for run in range(args.train_runs):
        print(f"\n--- Training Run {run + 1}/{args.train_runs} ---")
        env = GridWorldEnv(width=args.size, height=args.size, map_file=map_name)

        pi = train_policy(env, args.algo, render=args.render)
        if pi is None:
            print("Policy training skipped.")
            continue

        success = evaluate_single_episode(env, pi, render=args.render)
        result_text = "Success" if success else "Failure"
        print(f"Test Episode Result: {result_text}")
        success_count += int(success)

    print("\n=== Final Summary ===")
    print(f"{success_count}/{args.train_runs} episodes reached the goal.")
    print(f"Success Rate: {100.0 * success_count / args.train_runs:.1f}%")

if __name__ == "__main__":
    main()
