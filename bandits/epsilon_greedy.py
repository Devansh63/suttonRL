import numpy as np
import matplotlib.pyplot as plt

# multi armed bandit, epsilon greedy
# ch 2 of sutton and barto
# each arm has a true value drawn from N(0,1), actual rewards are N(q*, 1)

class Bandit:
    def __init__(self, k=10, epsilon=0.1):
        self.k = k
        self.epsilon = epsilon
        self.reset()

    def reset(self):
        self.q_true = np.random.randn(self.k)
        self.q_est = np.zeros(self.k)
        self.counts = np.zeros(self.k)
        self.total_reward = 0
        self.t = 0

    def choose(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k)
        return np.argmax(self.q_est)

    def step(self, action):
        reward = self.q_true[action] + np.random.randn()
        self.counts[action] += 1
        # incremental mean so we don't have to store all rewards
        self.q_est[action] += (reward - self.q_est[action]) / self.counts[action]
        self.total_reward += reward
        self.t += 1
        return reward


def run(epsilon, steps=1000, runs=200):
    rewards = np.zeros(steps)
    optimal = np.zeros(steps)

    for _ in range(runs):
        bandit = Bandit(epsilon=epsilon)
        best_arm = np.argmax(bandit.q_true)

        for t in range(steps):
            action = bandit.choose()
            reward = bandit.step(action)
            rewards[t] += reward
            if action == best_arm:
                optimal[t] += 1

    return rewards / runs, optimal / runs


if __name__ == "__main__":
    epsilons = [0, 0.01, 0.1]
    steps = 1000

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    for eps in epsilons:
        avg_rewards, pct_optimal = run(eps, steps=steps)
        label = f"e={eps}" if eps > 0 else "greedy (e=0)"
        ax1.plot(avg_rewards, label=label)
        ax2.plot(pct_optimal * 100, label=label)

    ax1.set_xlabel("steps")
    ax1.set_ylabel("avg reward")
    ax1.legend()
    ax1.set_title("10-armed bandit")

    ax2.set_xlabel("steps")
    ax2.set_ylabel("% optimal action")
    ax2.legend()
    ax2.set_title("% optimal action")

    plt.tight_layout()
    plt.savefig("bandits/bandit_results.png", dpi=120)
    plt.show()
    print("saved to bandits/bandit_results.png")
