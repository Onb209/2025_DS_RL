# # # algos/td_lambda_dqn.py
# #
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
from env.gridworld_c1 import GridWorldEnv_c1

# 한 번에 n-step 리턴을 계산하기 위한 Transition 타입
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class NStepReplayBuffer:
    """
    n-step 리턴을 계산하여 저장하는 리플레이 버퍼
    - n-step 버퍼(nstep_buf)로 최근 n개 transition 저장
    - 충분히 누적되면, 첫 transition에 대해 n-step 리턴을 계산해
      실제 버퍼(buffer)에 저장
    """
    def __init__(self, n: int, gamma: float, capacity: int = 10000):
        self.n = n                    # n-step 길이
        self.gamma = gamma            # 할인율
        self.buffer = deque(maxlen=capacity)  # 최종 저장 버퍼
        self.nstep_buf = deque()             # 임시 n-step 버퍼

    def push(self, state, action, reward, next_state, done):
        # 1) n-step 버퍼에 새 transition 추가
        self.nstep_buf.append((state, action, reward, next_state, done))
        # 2) 버퍼가 n개 모일 때까지 대기
        if len(self.nstep_buf) < self.n:
            return
        # 3) n-step 리턴 G 계산: sum_{k=0..n-1} γ^k * r_{t+k}
        G = 0.0
        for idx, (_, _, r, _, _) in enumerate(self.nstep_buf):
            G += (self.gamma ** idx) * r
        # 4) 저장할 transition 정보
        state0, action0, _, _, _ = self.nstep_buf[0]               # t 시점 state, action
        _, _, _, next_state_n, done_n = self.nstep_buf[-1]         # t+n 시점 next_state, done
        # 5) n-step 리턴을 사용해 최종 버퍼에 저장
        self.buffer.append(Transition(state0, action0, G, next_state_n, done_n))
        # 6) 가장 오래된 transition 제거 (슬라이딩 윈도우)
        self.nstep_buf.popleft()

    def sample(self, batch_size):
        # 무작위로 batch_size개 인덱스 샘플링
        idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idxs]
        # 배치 분리
        states, actions, rewards, next_states, dones = zip(*batch)

        # NumPy 배열로 변환
        states_np = np.stack(states).astype(np.float32)           # (B, state_dim)
        next_states_np = np.stack(next_states).astype(np.float32)
        actions_np = np.array(actions, dtype=np.int64)[:, None]    # (B,1)
        rewards_np = np.array(rewards, dtype=np.float32)[:, None]
        dones_np = np.array(dones, dtype=np.float32)[:, None]

        # PyTorch 텐서로 변환
        states_v = torch.from_numpy(states_np)
        next_states_v = torch.from_numpy(next_states_np)
        actions_v = torch.from_numpy(actions_np)
        rewards_v = torch.from_numpy(rewards_np)
        dones_v = torch.from_numpy(dones_np)

        return states_v, actions_v, rewards_v, next_states_v, dones_v

    def __len__(self):
        # 버퍼에 저장된 transition 수
        return len(self.buffer)


class QNetwork(nn.Module):
    """
    상태(state) → Q값(action_dim) 예측 신경망
    - 마지막 bias를 init_q로 상수 초기화하여
      낙관적 초기화(optimistic init) 가능
    """
    def __init__(self,
                 state_dim: int = 2,
                 action_dim: int = 8,
                 hidden_dim: int = 256,
                 init_q: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        # 마지막 레이어 bias를 높은 값으로 초기화하면
        # 방문 전까지 Q값이 높다고 가정 → 탐험 촉진
        nn.init.constant_(self.net[-1].bias, init_q)

    def forward(self, x):
        # x: (B, state_dim) → 출력: (B, action_dim)
        return self.net(x)


class DQNAgent:
    """
    DQN 에이전트
    - n-step return 사용
    - target network와 soft update (τ)
    - episode 기반 ε 스케줄: warm-up 후 선형 감쇠
    """
    def __init__(
        self,
        env: GridWorldEnv_c1,
        n_step: int = 3,
        gamma: float = 0.999,
        lr: float = 1e-3,
        max_episodes: int = 2000,
        warmup_episodes: int = 1000,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update_steps: int = 1000,
        tau: float = 0.005,
        device: torch.device = None
    ):
        # 기본 설정
        self.env = env
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gamma = gamma
        self.n_step = n_step
        self.batch_size = batch_size

        # soft update 비율
        self.tau = tau
        self.target_update_steps = target_update_steps
        self.steps_done = 0

        # ε-greedy 파라미터
        self.max_episodes = max_episodes
        self.warmup_episodes = warmup_episodes
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon = epsilon_start

        # Q-network, target network, optimizer
        self.qnet = QNetwork().to(self.device)
        self.target_net = QNetwork().to(self.device)
        self.target_net.load_state_dict(self.qnet.state_dict())
        self.target_net.eval()  # target network는 학습 비활성화
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr)

        # n-step 리플레이 버퍼
        self.replay = NStepReplayBuffer(n_step, gamma, capacity=buffer_size)

    def select_action(self, state: np.ndarray, eval: bool = False) -> int:
        """
        ε-greedy 행동 선택
        - eval=True: ε=0 (순수 greedy)
        """
        eps = 0.0 if eval else self.epsilon
        if not eval and np.random.rand() < eps:
            # 탐험: 랜덤 행동
            return np.random.randint(0, 8)
        # 활용: 네트워크에서 max Q 선택
        state_v = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(self.qnet(state_v).argmax(dim=1).item())

    def learn(self,
              state: np.ndarray,
              action: int,
              reward: float,
              next_state: np.ndarray,
              done: bool) -> float:
        """
        한 스텝마다 호출:
        - 리플레이 버퍼에 (n-step) transition 저장
        - 버퍼가 가득 차면 배치 학습
        - soft update로 target network 동기화
        """
        # 리플레이 버퍼에 저장 (n-step 자동 처리)
        self.replay.push(state, action, reward, next_state, done)
        loss = 0.0

        # 충분한 데이터가 쌓이면 학습 수행
        if len(self.replay) >= self.batch_size:
            # 배치 샘플
            states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)
            states, actions = states.to(self.device), actions.to(self.device)
            rewards, next_states = rewards.to(self.device), next_states.to(self.device)
            dones = dones.to(self.device)

            # 현재 Q값
            q_values = self.qnet(states).gather(1, actions)
            # target Q: r + γ^n * max_a' Q_target(next_states, a')
            with torch.no_grad():
                q_next = self.target_net(next_states).max(1)[0].unsqueeze(1)
                q_target = rewards + (self.gamma ** self.n_step) * q_next * (1 - dones)

            # MSE loss & 최적화
            loss = nn.functional.mse_loss(q_values, q_target)
            self.optimizer.zero_grad()
            loss.backward()
            # gradient clipping (안정성 위해)
            nn.utils.clip_grad_norm_(self.qnet.parameters(), max_norm=1.0)
            self.optimizer.step()

            # soft update: θ_target ← τ θ + (1-τ) θ_target
            self.steps_done += 1
            if self.steps_done % self.target_update_steps == 0:
                for p_t, p in zip(self.target_net.parameters(), self.qnet.parameters()):
                    p_t.data.copy_(self.tau * p.data + (1 - self.tau) * p_t.data)

        return loss

    def reset_episode(self):
        # 특별히 에피소드 단위로 초기화할 것은 없음
        pass

    def finish_episode(self, episode_idx: int) -> None:
        """
        에피소드가 끝날 때마다 ε 스케줄 업데이트
        - warm-up 구간 동안 ε 유지
        - 이후 선형적으로 ε_min까지 감소
        """
        if episode_idx < self.warmup_episodes:
            self.epsilon = self.epsilon_start
        else:
            frac = (episode_idx - self.warmup_episodes) / max(1, self.max_episodes - self.warmup_episodes)
            self.epsilon = max(
                self.epsilon_min,
                self.epsilon_start - frac * (self.epsilon_start - self.epsilon_min)
            )

    def save(self, path: str = 'checkpoints/dqn.pth') -> None:
        """네트워크 파라미터 저장"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.qnet.state_dict(), path)

    def load(self, path: str) -> None:
        """저장된 파라미터 불러오기 및 target network 동기화"""
        self.qnet.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.qnet.state_dict())