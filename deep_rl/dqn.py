import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque
import random

# DQN on CartPole-v1
# replay buffer + target network like the deepmind paper (Mnih 2015)
# target net makes training way more stable, without it Q values explode

BUFFER_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-3
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE = 10


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones)
        )

    def __len__(self):
        return len(self.buffer)


def train(episodes=400):
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    policy_net = QNetwork(state_dim, action_dim).to(device)
    target_net = QNetwork(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    buffer = ReplayBuffer(BUFFER_SIZE)

    epsilon = EPSILON_START
    rewards_per_ep = []

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    s = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action = policy_net(s).argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.push(state, action, reward, next_state, float(done))

            state = next_state
            total_reward += reward

            if len(buffer) >= BATCH_SIZE:
                states, actions, rewards_b, next_states, dones = buffer.sample(BATCH_SIZE)
                states = states.to(device)
                actions = actions.to(device)
                rewards_b = rewards_b.to(device)
                next_states = next_states.to(device)
                dones = dones.to(device)

                current_q = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
                with torch.no_grad():
                    max_next_q = target_net(next_states).max(1)[0]
                    target_q = rewards_b + GAMMA * max_next_q * (1 - dones)

                loss = nn.MSELoss()(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        rewards_per_ep.append(total_reward)

        if ep % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if ep % 50 == 0:
            print(f"ep {ep} | avg (last 20): {np.mean(rewards_per_ep[-20:]):.1f} | eps: {epsilon:.3f}")

    env.close()
    return rewards_per_ep


if __name__ == "__main__":
    rewards = train()

    smoothed = np.convolve(rewards, np.ones(20)/20, mode="valid")

    plt.figure(figsize=(10, 4))
    plt.plot(rewards, alpha=0.3, label="raw", color="green")
    plt.plot(smoothed, label="smoothed (20 ep)", color="green")
    plt.axhline(y=475, color="red", linestyle="--", alpha=0.5, label="solved threshold")
    plt.xlabel("episode")
    plt.ylabel("total reward")
    plt.title("DQN on CartPole-v1")
    plt.legend()
    plt.tight_layout()
    plt.savefig("deep_rl/dqn_results.png", dpi=120)
    plt.show()
