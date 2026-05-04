import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

# value iteration on FrozenLake
# ch 4, needs the full model (transition probs)
# gymnasium gives us env.unwrapped.P[state][action] = [(prob, next_state, reward, done)]

def value_iteration(env, gamma=0.99, theta=1e-6):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    V = np.zeros(n_states)
    P = env.unwrapped.P

    iters = 0
    while True:
        delta = 0
        for s in range(n_states):
            v = V[s]
            action_vals = []
            for a in range(n_actions):
                val = sum(prob * (r + gamma * V[ns]) for prob, ns, r, _ in P[s][a])
                action_vals.append(val)
            V[s] = max(action_vals)
            delta = max(delta, abs(v - V[s]))
        iters += 1
        if delta < theta:
            break

    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        action_vals = []
        for a in range(n_actions):
            val = sum(prob * (r + gamma * V[ns]) for prob, ns, r, _ in P[s][a])
            action_vals.append(val)
        policy[s] = np.argmax(action_vals)

    print(f"converged in {iters} iterations")
    return V, policy


def evaluate(env, policy, episodes=100):
    wins = 0
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = policy[state]
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        if reward == 1.0:
            wins += 1
    return wins / episodes


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=True)

    V, policy = value_iteration(env)
    win_rate = evaluate(env, policy)
    print(f"win rate: {win_rate*100:.1f}%")

    grid = V.reshape(4, 4)
    plt.figure(figsize=(6, 5))
    plt.imshow(grid, cmap="Blues")
    for i in range(4):
        for j in range(4):
            plt.text(j, i, f"{grid[i,j]:.2f}", ha="center", va="center", fontsize=9)
    plt.title("value function - FrozenLake (value iteration)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("dynamic_programming/value_function.png", dpi=120)
    plt.show()

    env.close()
