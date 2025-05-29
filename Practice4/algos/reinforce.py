import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

class PolicyNetwork(nn.Module):
    """
    정책 신경망: 주어진 상태(state)를 입력받아
    행동(action)의 평균(mean)과 표준편차(std)를 출력합니다.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(PolicyNetwork, self).__init__()
        # 은닉층 구성: state_dim -> hidden_dim -> hidden_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # 출력층: hidden_dim -> action_dim (평균)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        # 행동 분포의 로그 표준편차를 학습 가능한 파라미터로 선언
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        순전파 함수:
        state shape: [batch_size, state_dim]
        반환: mean shape [batch_size, action_dim], std shape [action_dim]
        """
        x = self.net(state)
        mean = self.mean_layer(x)
        std = torch.exp(self.log_std)  # 양수 표준편차
        return mean, std

class REINFORCEAgent:
    """
    REINFORCE 에이전트:
    - 에피소드별 log_prob과 reward 버퍼를 유지
    - 에피소드 종료 시 정책 파라미터 업데이트
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        device: torch.device = torch.device('cpu')
    ):
        """
        초기화 파라미터:
        state_dim: 상태 차원 수
        action_dim: 행동 차원 수
        lr: 학습률
        gamma: 할인율
        device: 연산 디바이스 ("cpu" 또는 "cuda")
        """
        self.device = device
        # 정책 네트워크 생성 및 디바이스 할당
        self.policy = PolicyNetwork(state_dim, action_dim).to(self.device)
        # 최적화 기법: Adam
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma

        # 에피소드별 버퍼 초기화
        self.log_probs: list[torch.Tensor] = []  # 행동 선택 시 log_prob 저장
        self.rewards: list[float] = []         # 에피소드 보상 저장

    def reset_episode(self) -> None:
        """새 에피소드 시작 시 버퍼 비우기"""
        self.log_probs.clear()
        self.rewards.clear()

    def select_action(self, state: torch.Tensor, eval_mode: bool = False) -> torch.Tensor:
        """
        상태(state)를 받아 행동(action)을 샘플링하거나 선택합니다.
        eval_mode=True일 경우 평균 행동(mean)을 반환합니다.

        Args:
            state: torch.Tensor, shape [state_dim]
            eval_mode: bool, 평가 모드 여부
        Returns:
            action: torch.Tensor, shape [action_dim]
        """
        state = state.to(self.device).unsqueeze(0)  # 배치 차원 추가
        mean, std = self.policy(state)
        if eval_mode:
            # 평가 시에는 평균 행동 사용 (클램프로 범위 제한)
            action = torch.clamp(mean, -1.0, 1.0)
            return action.squeeze(0).cpu()

        # 확률 분포 생성 및 reparameterization 샘플링
        dist = Normal(mean, std)
        action = dist.rsample()  # 재파라미터화
        # log_prob 저장 (after rsample)
        log_prob = dist.log_prob(action).sum(dim=-1)
        self.log_probs.append(log_prob)
        # 실제 환경에 보낼 행동값은 클램프 처리
        action = torch.clamp(action, -1.0, 1.0)
        return action.squeeze(0).cpu()

    def finish_episode(self) -> float:
        """
        에피소드가 종료된 후 호출하여:
        1) 할인 누적 보상 계산
        2) 보상 정규화
        3) 정책 손실(policy loss) 계산 및 역전파
        4) 정책 네트워크 업데이트
        5) 버퍼 초기화

        Returns:
            loss.item(): 업데이트 후 손실 값 (float)
        """
        # 1) 할인 누적 보상 계산 (G_t)
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # 2) 보상 정규화 (선택적)
        returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-9)

        # 3) 정책 손실: -sum(log_prob * return)
        policy_loss = [-lp * Gt for lp, Gt in zip(self.log_probs, returns)]
        loss = torch.stack(policy_loss).sum()

        # 4) 역전파 및 파라미터 업데이트
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 5) 다음 에피소드 준비를 위한 버퍼 초기화
        self.reset_episode()

        return loss.item()

    def save(self, path: str) -> None:
        """정책 네트워크 파라미터를 파일에 저장"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str) -> None:
        """저장된 파라미터를 불러와 정책 네트워크에 로드"""
        self.policy.load_state_dict(torch.load(path, map_location=self.device))

    def inference(self, state: torch.Tensor) -> torch.Tensor:
        """
        평가 모드에서 행동(action)을 반환합니다.
        내부적으로 select_action(eval_mode=True) 호출
        """
        return self.select_action(state, eval_mode=True)