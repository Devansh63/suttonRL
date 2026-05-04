import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt

# REINFORCE on CartPole-v1
# ch 13, simplest policy gradient method
# high variance compared to actor-critic but no critic needed
# normalizing returns per episode helps a lot with stability

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)


def train(episodes=1000, gamma=0.99, lr=1e-3):
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNet(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    rewards_per_ep = []

    for ep in range(episodes):
        state, _ = env.reset()
        log_probs = []
        rewards = []
        done = False

        while not done:
            state_t = torch.FloatTensor(state)
            probs = policy(state_t)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))

            state, reward, terminated, truncated, _ = env.step(action.item())
            rewards.append(reward)
            done = terminated or truncated

        # compute discounted returns
        G = 0
        returns = []
        for r in reversed(rewards):
            G = gamma * G + r
            returns.insert(0, G)

        returns = torch.FloatTensor(returns)
        # normalize to reduce variance, makes training much more stable
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        log_probs = torch.stack(log_probs)
        loss = -(log_probs * returns).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rewards_per_ep.append(sum(rewards))

        if ep % 100 == 0:
            print(f"ep {ep} | avg (last 20): {np.mean(rewards_per_ep[-20:]):.1f}")

    env.close()
    return rewards_per_ep


if __name__ == "__main__":
    rewards = train(episodes=1000)

    smoothed = np.convolve(rewards, np.ones(30)/30, mode="valid")

    plt.figure(figsize=(10, 4))
    plt.plot(rewards, alpha=0.3, color="purple", label="raw")
    plt.plot(smoothed, color="purple", label="smoothed (30 ep)")
    plt.axhline(y=475, color="red", linestyle="--", alpha=0.5, label="solved")
    plt.xlabel("episode")
    plt.ylabel("total reward")
    plt.title("REINFORCE on CartPole-v1")
    plt.legend()
    plt.tight_layout()
    plt.savefig("policy_gradient/reinforce_results.png", dpi=120)
    plt.show()
