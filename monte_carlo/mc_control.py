import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from collections import defaultdict

# first-visit MC control on Blackjack-v1
# ch 5, on-policy version with epsilon-soft
# takes way more episodes than TD methods to converge but doesn't need a model
# blackjack is nice here because episodes are naturally short

def get_action(Q, state, epsilon, n_actions):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    return np.argmax(Q[state])


def train(episodes=500000, epsilon=0.1, gamma=1.0):
    env = gym.make("Blackjack-v1")
    n_actions = env.action_space.n

    Q = defaultdict(lambda: np.zeros(n_actions))
    returns = defaultdict(list)

    for ep in range(episodes):
        trajectory = []
        state, _ = env.reset()
        done = False

        while not done:
            action = get_action(Q, state, epsilon, n_actions)
            next_state, reward, terminated, truncated, _ = env.step(action)
            trajectory.append((state, action, reward))
            state = next_state
            done = terminated or truncated

        # first-visit: only update the first time we see each (s, a) pair
        visited = set()
        G = 0
        for s, a, r in reversed(trajectory):
            G = gamma * G + r
            if (s, a) not in visited:
                visited.add((s, a))
                returns[(s, a)].append(G)
                Q[s][a] = np.mean(returns[(s, a)])

        if ep % 100000 == 0:
            print(f"ep {ep}")

    env.close()
    return Q


def eval_greedy(Q, episodes=10000):
    env = gym.make("Blackjack-v1")
    wins = 0
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = np.argmax(Q[state]) if state in Q else 0
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        if reward > 0:
            wins += 1
    env.close()
    return wins / episodes


if __name__ == "__main__":
    print("training... this takes a while")
    Q = train(episodes=500000)

    win_rate = eval_greedy(Q)
    print(f"win rate (greedy policy): {win_rate*100:.1f}%")

    # plot value function for no usable ace states
    player_sums = range(12, 22)
    dealer_cards = range(1, 11)

    V = np.zeros((10, 10))
    for i, ps in enumerate(player_sums):
        for j, dc in enumerate(dealer_cards):
            state = (ps, dc, False)
            V[i, j] = np.max(Q[state]) if state in Q else 0

    plt.figure(figsize=(8, 6))
    plt.imshow(V, origin="lower", cmap="RdYlGn", aspect="auto")
    plt.colorbar(label="value")
    plt.xlabel("dealer showing")
    plt.ylabel("player sum")
    plt.xticks(range(10), range(1, 11))
    plt.yticks(range(10), range(12, 22))
    plt.title("MC control value function (no usable ace)")
    plt.tight_layout()
    plt.savefig("monte_carlo/blackjack_values.png", dpi=120)
    plt.show()
