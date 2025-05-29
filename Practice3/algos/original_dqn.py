import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple

from env.gridworld_c1 import GridWorldEnv_c1

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class QNetwork(nn.Module):
    def __init__(self, state_dim=2, action_dim=8, hidden_dim=256):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size):
        idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idxs]
        states, actions, rewards, next_states, dones = zip(*batch)
        states_np = np.stack(states).astype(np.float32)
        next_states_np = np.stack(next_states).astype(np.float32)
        actions_np = np.array(actions, dtype=np.int64)[:, None]
        rewards_np = np.array(rewards, dtype=np.float32)[:, None]
        dones_np = np.array(dones, dtype=np.float32)[:, None]
        return (
            torch.from_numpy(states_np),
            torch.from_numpy(actions_np),
            torch.from_numpy(rewards_np),
            torch.from_numpy(next_states_np),
            torch.from_numpy(dones_np)
        )

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """
    Original DQN agent with state/reward normalization, MSE loss, and per-step epsilon decay.
    """
    def __init__(
        self,
        env: GridWorldEnv_c1,
        lr: float = 1e-3,
        gamma: float = 0.995,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.1,
        epsilon_decay: float = 0.9995,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update: int = 1000,
        device: torch.device = None
    ):
        self.env = env
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update

        # normalization scales
        self.state_scale = np.array([env.height, env.width], dtype=np.float32)
        self.reward_scale = 100.0

        # epsilon-greedy
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

        # networks
        self.qnet = QNetwork().to(self.device)
        self.target_net = QNetwork().to(self.device)
        self.target_net.load_state_dict(self.qnet.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr)

        # replay
        self.memory = ReplayBuffer(buffer_size)

    def select_action(self, state, eval=False):
        s = np.array(state, dtype=np.float32) / self.state_scale
        state_v = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
        if not eval and np.random.rand() < self.epsilon:
            return np.random.randint(0, 8)
        with torch.no_grad():
            return int(self.qnet(state_v).argmax(dim=1).item())

    def learn(self, state, action, reward, next_state, next_action=None, done=False):
        s = np.array(state, dtype=np.float32) / self.state_scale
        ns = np.array(next_state, dtype=np.float32) / self.state_scale
        r = reward / self.reward_scale
        self.memory.push(s, action, r, ns, float(done))
        if len(self.memory) < self.batch_size:
            return None
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # current Q
        q_values = self.qnet(states).gather(1, actions)

        # original DQN target
        with torch.no_grad():
            q_next = self.target_net(next_states).max(1)[0].unsqueeze(1)
            q_target = rewards + self.gamma * q_next * (1 - dones)

        # MSE loss
        loss = nn.functional.mse_loss(q_values, q_target)

        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), max_norm=1.0)
        self.optimizer.step()

        # update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

        # update target network
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.qnet.state_dict())
        return loss.item()

    def reset_episode(self):
        pass

    def finish_episode(self, episode_idx):
        pass

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.qnet.state_dict(), path)

    def load(self, path):
        self.qnet.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.qnet.state_dict())

    def inference(self, state):
        return self.select_action(state, eval=True)


