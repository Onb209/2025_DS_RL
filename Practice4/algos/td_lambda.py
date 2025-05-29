import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from env.gridworld_c1 import GridWorldEnv_c1

class QNetwork(nn.Module):
    """
    Fully-connected Q-network: state_dim 크기의 state를 받아 action_dim 개수의 Q-value를 출력하는 MLP.
    - state_dim: 입력 state 차원 (예: (row, col) → 2)
    - action_dim: 행동 개수 (예: 8)
    - hidden_dim: hidden layer 유닛 수
    - init_q: 마지막 bias를 init_q로 초기화하여 optimistic init 구현
    """
    def __init__(self, state_dim=2, action_dim=8, hidden_dim=256, init_q: float = 0.0):
        super().__init__()
        # 3-layer MLP 구성
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        # 마지막 layer bias를 init_q로 설정 (optimistic init)
        nn.init.constant_(self.net[-1].bias, init_q)

    def forward(self, x):
        # Q(s,a) 계산
        return self.net(x)

class TDLambdaDQNAgent:
    """
    TD(λ)-DQN agent (forward-view λ-return 기반 episodic update)

    - 에피소드 동안 (state, action, reward) sequence를 저장
    - 에피소드 종료 시 각 timestep에 대해 λ-return 계산
    - 단일 batch update로 Q-network 학습 (MSE loss)
    - gradient clipping 적용으로 안정성 확보
    - ε-greedy policy 에피소드 단위 decay
    """
    def __init__(
        self,
        env: GridWorldEnv_c1,
        lr: float = 1e-3,
        gamma: float = 0.99,
        lambd: float = 0.8,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.1,
        epsilon_decay: float = 0.995,
        device: torch.device = None
    ):
        self.env = env
        self.device = device or torch.device('cpu')
        # Discount factor (γ)
        self.gamma = gamma
        # eligibility trace decay 파라미터 λ
        self.lambd = lambd

        # ε-greedy 파라미터
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-network 및 optimizer 초기화
        self.qnet = QNetwork().to(self.device)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr)

        # 에피소드 버퍼: (state, action, reward) 리스트
        self.episode = []

    def select_action(self, state: np.ndarray, eval: bool = False) -> int:
        """
        ε-greedy 행동 선택
        - eval=True: ε=0 (greedy)
        - eval=False: 확률 ε로 random, 아니면 greedy
        """
        eps = 0.0 if eval else self.epsilon
        if not eval and np.random.rand() < eps:
            # 랜덤 탐험
            return np.random.randint(0, len(self.env.deltas))
        # greedy 선택
        state_v = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(self.qnet(state_v).argmax(dim=1).item())

    def learn(self, state, action, reward, next_state, next_action=None, done=False):
        """
        각 스텝에서 호출: transition을 버퍼에 저장
        Q-network update는 에피소드 끝에서 수행
        """
        # state 복사하여 버퍼에 추가
        self.episode.append((state.copy(), action, reward))
        return None

    def reset_episode(self):
        """
        새로운 에피소드 시작 전 버퍼 초기화
        """
        self.episode.clear()

    def finish_episode(self) -> float:
        """
        에피소드 종료 시 호출
        - forward-view λ-return 계산
        - MSE loss로 Q-network update
        - ε decay 적용
        - episode buffer clear
        """
        T = len(self.episode)
        if T == 0:
            return 0.0

        # 버퍼에서 상태, 행동, 보상 분리
        states, actions, rewards = zip(*self.episode)
        # 텐서 변환
        states_v = torch.tensor(np.stack(states), dtype=torch.float32, device=self.device)
        actions_v = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)

        # all Q-values 계산 (bootstrapping 용)
        q_all = self.qnet(states_v).cpu().detach().numpy()  # shape (T, A)

        # forward-view λ-return 계산
        lambda_returns = []
        for t in range(T):
            G_lambda = 0.0
            coeff = 1.0 - self.lambd
            # 1-step 부터 (T-t-1)-step까지
            for n in range(1, T - t):
                # n-step return G^{(n)}_t
                G_n = sum((self.gamma ** k) * rewards[t + k] for k in range(n))
                # bootstrap term Q(s_{t+n})
                G_n += (self.gamma ** n) * np.max(q_all[t + n])
                G_lambda += coeff * G_n
                coeff *= self.lambd
            # Monte Carlo term
            G_T = sum((self.gamma ** k) * rewards[t + k] for k in range(T - t))
            G_lambda += (self.lambd ** (T - t)) * G_T

            lambda_returns.append(G_lambda)

        # λ-returns → tensor targets
        targets_v = torch.tensor(lambda_returns, dtype=torch.float32, device=self.device).unsqueeze(1)

        # 실제 예측 Q-pred
        q_pred = self.qnet(states_v).gather(1, actions_v)

        # MSE loss
        loss = nn.functional.mse_loss(q_pred, targets_v)

        # 역전파 및 최적화
        self.optimizer.zero_grad()
        loss.backward()
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), max_norm=1.0)
        self.optimizer.step()

        # ε decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

        # 버퍼 초기화
        self.episode.clear()
        return loss.item()

    def save(self, path: str) -> None:
        """
        Q-network weights 저장
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.qnet.state_dict(), path)

    def load(self, path: str) -> None:
        """
        저장된 weights 로드
        """
        sd = torch.load(path, map_location=self.device)
        self.qnet.load_state_dict(sd)