import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from env.gridworld_c1 import GridWorldEnv_c1

class QNetwork(nn.Module):
    """
    상태(state) 벡터를 입력받아, 각 action에 대한 Q-value를 출력하는 신경망입니다.
    - 입력 크기: state_dim (기본 2: row, col)
    - 출력 크기: action_dim (기본 8 방향)
    """
    def __init__(self, state_dim=2, action_dim=8, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        # 순전파: 배치 형태 (B, state_dim) → (B, action_dim)
        return self.net(x)


class DeepSARSAAgent:
    """
    Deep SARSA 에이전트
    - on-policy 업데이트 (SARSA)
    - 상태 정규화(state normalization)
    - step 기반 epsilon 감쇠(epsilon decay)
    """
    def __init__(
        self,
        env: GridWorldEnv_c1,
        lr: float = 1e-4,
        gamma: float = 0.999,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.9999,
        device: torch.device = None
    ):
        # 환경 및 장치 설정
        self.env = env
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gamma = gamma  # 할인율

        # ε-greedy 파라미터 초기화
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.total_steps = 0  # 학습 스텝 카운터

        # Q-network 및 최적화기 설정
        self.qnet = QNetwork().to(self.device)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr)

    def select_action(self, state: np.ndarray, eval: bool = False) -> int:
        """
        행동 선택 함수
        - eval=True: 탐험 없이 greedy (최댓값 행동)
        - eval=False: ε 확률로 랜덤, 나머지 greedy
        """
        # 상태 정규화: [0,1] 범위로 스케일
        state_norm = state.astype(np.float32) / np.array([self.env.height, self.env.width], np.float32)
        state_v = torch.tensor(state_norm, dtype=torch.float32, device=self.device).unsqueeze(0)

        if eval:
            # 평가 모드: 항상 greedy
            with torch.no_grad():
                return int(self.qnet(state_v).argmax(dim=1).item())
        else:
            # 학습 모드: ε-greedy
            if np.random.rand() < self.epsilon:
                # 탐험
                return np.random.randint(0, 8)
            # 활용
            with torch.no_grad():
                return int(self.qnet(state_v).argmax(dim=1).item())

    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_action: int,
        done: bool
    ) -> float:
        """
        Deep SARSA 업데이트 수행
        - Q(s,a) 값과 목표(target)를 계산하여 MSE loss로 학습
        - 그 후 ε를 decay 비율만큼 감소
        """
        # 상태 정규화 후 텐서 변환
        state_norm = state.astype(np.float32) / np.array([self.env.height, self.env.width], np.float32)
        next_state_norm = next_state.astype(np.float32) / np.array([self.env.height, self.env.width], np.float32)
        state_v = torch.tensor(state_norm, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_state_v = torch.tensor(next_state_norm, dtype=torch.float32, device=self.device).unsqueeze(0)

        # 현재 Q(s,a)
        q_val = self.qnet(state_v)[0, action]

        # SARSA target: r + γ * Q(s', a') (만약 에피소드 종료라면 γ * Q = 0)
        with torch.no_grad():
            q_next = self.qnet(next_state_v)[0, next_action]
            target = reward + (0.0 if done else self.gamma * q_next)

        # MSE loss
        loss = (q_val - target).pow(2).mean()

        # 역전파 및 최적화
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ε 감쇠 (step 기반)
        self.total_steps += 1
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

        return loss.item()

    def reset_episode(self):
        """
        에피소드 단위로 초기화할 것이 없다면 pass
        (ε는 학습 스텝마다 decay됨)
        """
        pass

    def finish_episode(self, episode_idx: int = None) -> None:
        """
        에피소드 종료 시 추가 작업이 필요 없으므로 No-op
        """
        return

    def save(self, path: str) -> None:
        """
        model과 optimizer 상태를 파일로 저장
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict':     self.qnet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        """
        저장된 checkpoint를 불러와 network와 optimizer에 로드
        """
        ckpt = torch.load(path, map_location=self.device)
        self.qnet.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    def inference(self, state: np.ndarray) -> int:
        """
        평가 모드에서 greedy 행동 반환
        """
        return self.select_action(state, eval=True)