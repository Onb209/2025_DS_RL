import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from env.gridworld_c2 import GridWorldEnv_c2

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim=2, action_dim=2, hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # learnable log std

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_layer(x)
        std = torch.exp(self.log_std) + 1e-6  # ensure positive
        return mean, std

class REINFORCEAgent:
    def __init__(self,
                 env: GridWorldEnv_c2,
                 lr=1e-3,
                 gamma=0.999,
                 entropy_coeff=0.01):
        self.env = env
        self.device = torch.device('cpu')
        self.policy = PolicyNetwork().to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff

        self.log_probs = []
        self.rewards = []
        self.entropies = []  # store entropy per step

    def reset_episode(self):
        self.log_probs = []
        self.rewards = []
        self.entropies = []

    def select_action(self, state, eval=False):
        state_v = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        mean, std = self.policy(state_v)
        if eval:
            action = torch.clamp(mean, -1.0, 1.0)
            return action.cpu().detach().numpy()[0]

        dist = Normal(mean, std)
        action = dist.rsample()  # use rsample for reparameterization
        action = torch.clamp(action, -1.0, 1.0)

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        self.log_probs.append(log_prob)
        self.entropies.append(entropy)
        return action.cpu().detach().numpy()[0]

    def learn(self, state, action, reward, next_state, next_action=None, done=False):
        # accumulate rewards only; learning in finish_episode
        self.rewards.append(reward)
        return None

    def finish_episode(self):
        # compute discounted returns
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-9)

        # policy loss
        policy_losses = []
        for log_prob, Gt in zip(self.log_probs, returns):
            policy_losses.append(-log_prob * Gt)
        # entropy bonus (maximize entropy)
        entropy_bonus = 0
        for ent in self.entropies:
            entropy_bonus += ent
        entropy_loss = -self.entropy_coeff * entropy_bonus

        loss = torch.stack(policy_losses).sum() + entropy_loss

        # update policy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))

    def inference(self, state):
        return self.select_action(state, eval=True)
