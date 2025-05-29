import argparse
import importlib
import os
from env import gridworld_d2, gridworld_c2

# 알고리즘 이름과 클래스 이름 매핑
CLASS_MAP = {
    'deepsarsa': 'DeepSARSAAgent',
    'dqn':       'DQNAgent',
    'reinforce': 'REINFORCEAgent',
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', choices=list(CLASS_MAP.keys()), required=True)
    parser.add_argument('--map', type=str, default='map1.yaml')
    parser.add_argument('--iter', type=int, default=10)
    args = parser.parse_args()

    # 설정 파일 경로
    config_path = os.path.join('configs', args.map)

    # 모듈 및 에이전트 클래스 동적 로드
    mod = importlib.import_module(f'algos.{args.algo}')
    AgentClass = getattr(mod, CLASS_MAP[args.algo])

    # 환경 초기화
    if args.algo == 'reinforce':
        env = gridworld_c2.GridWorldEnv_c2(config_path)
    else:
        env = gridworld_d2.GridWorldEnvDiscrete(config_path)

    # 에이전트 생성 및 체크포인트 로드
    agent = AgentClass(env)
    os.makedirs('checkpoints', exist_ok=True)
    agent.load(os.path.join('checkpoints', f'{args.algo}.pth'))

    # 테스트 실행
    for _ in range(args.iter):
        state = env.reset()
        done = False
        while not done:
            # eval=True 로 탐험 없이 결정적 행동 수행
            if args.algo == 'reinforce':
                action = agent.inference(state)
            else:
                action = agent.select_action(state, eval=True)
            state, reward, done, _ = env.step(action)
            env.render()
    env.close()

if __name__ == '__main__':
    main()