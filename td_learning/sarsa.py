import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

# SARSA on CliffWalking
# on-policy TD control, ch 6.4
# key thing: next action is chosen before the update, so the update uses
# what the policy would actually do, not the greedy max

def train(env, episodes=500, alpha=0.5, gamma=0.99, epsilon=0.1):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))

    rewards_per_ep = []

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if np.random.rand() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(Q[next_state])

            Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])

            state = next_state
            action = next_action
            total_reward += reward

        rewards_per_ep.append(total_reward)

    return Q, rewards_per_ep


if __name__ == "__main__":
    env = gym.make("CliffWalking-v0")
    Q, rewards = train(env, episodes=500)

    smoothed = np.convolve(rewards, np.ones(20)/20, mode="valid")

    plt.figure(figsize=(10, 4))
    plt.plot(rewards, alpha=0.3, color="orange", label="raw")
    plt.plot(smoothed, color="orange", label="smoothed (20 ep)")
    plt.xlabel("episode")
    plt.ylabel("total reward")
    plt.title("SARSA on CliffWalking")
    plt.legend()
    plt.tight_layout()
    plt.savefig("td_learning/sarsa_results.png", dpi=120)
    plt.show()

    env.close()
    print(f"final avg reward (last 50 eps): {np.mean(rewards[-50:]):.1f}")
