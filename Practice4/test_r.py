import os
import argparse
import torch
from env.gridworld_c2 import GridWorldEnv_c2
from algos.reinforce import REINFORCEAgent


def main():
    parser = argparse.ArgumentParser(description='Test trained REINFORCE policy')
    parser.add_argument('--map', type=str, default='map1.yaml', help='Map configuration file')
    parser.add_argument('--model', type=str, default='reinforce.pth', help='Path to trained model checkpoint')
    parser.add_argument('--attempts', type=int, default=10, help='Number of test episodes')
    parser.add_argument('--max-steps', type=int, default=300, help='Maximum steps per episode')
    args = parser.parse_args()

    # 환경 초기화
    config_path = os.path.join('configs', args.map)
    env = GridWorldEnv_c2(config_path)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # 에이전트 초기화 및 모델 로드
    device = torch.device('cpu')
    agent = REINFORCEAgent(state_dim=state_dim, action_dim=action_dim, device=device)
    model_path = os.path.join('checkpoints', args.model)
    agent.load(model_path)

    map_name = os.path.basename(args.map)
    successes = 0
    failures = 0

    for ep in range(1, args.attempts + 1):
        state = env.reset()
        done = False
        reward_accum = 0.0

        for step in range(args.max_steps):
            # 상태를 텐서로 변환하여 결정론적 행동 선택
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            action_tensor = agent.inference(state_tensor)
            action = action_tensor.detach().cpu().numpy()

            next_state, reward, done, _ = env.step(action)
            reward_accum += reward
            state = next_state

            env.render()

            if done:
                break

        # 목표 달성 여부 판단 (보상 100 받으면 성공)
        if reward_accum >= 100.0:
            successes += 1
        else:
            failures += 1

        print(f"Episode {ep}: Reward {reward_accum:.2f}, {'Success' if reward_accum >= 100.0 else 'Failure'}")

    env.close()
    success_rate = 100.0 * successes / args.attempts if args.attempts > 0 else 0.0

    # 최종 결과 출력
    print('--- Test Summary ---')
    print(f"Map: {map_name}")
    print(f"Attempts: {args.attempts}")
    print(f"Successes: {successes}")
    print(f"Failures: {failures}")
    print(f"Success Rate: {success_rate:.2f}%")


if __name__ == '__main__':
    main()