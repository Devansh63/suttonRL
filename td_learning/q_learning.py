import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

# Q-learning on CliffWalking
# off-policy TD control, ch 6.5
# unlike SARSA, update always uses max Q for next state
# so it learns optimal policy even while behaving epsilon-greedily

def train(env, episodes=500, alpha=0.5, gamma=0.99, epsilon=0.1):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))

    rewards_per_ep = []

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            best_next = np.max(Q[next_state])
            Q[state, action] += alpha * (reward + gamma * best_next - Q[state, action])

            state = next_state
            total_reward += reward

        rewards_per_ep.append(total_reward)

    return Q, rewards_per_ep


if __name__ == "__main__":
    env = gym.make("CliffWalking-v0")
    Q, rewards = train(env, episodes=500)

    smoothed = np.convolve(rewards, np.ones(20)/20, mode="valid")

    plt.figure(figsize=(10, 4))
    plt.plot(rewards, alpha=0.3, color="blue", label="raw")
    plt.plot(smoothed, color="blue", label="smoothed (20 ep)")
    plt.xlabel("episode")
    plt.ylabel("total reward")
    plt.title("Q-Learning on CliffWalking")
    plt.legend()
    plt.tight_layout()
    plt.savefig("td_learning/q_learning_results.png", dpi=120)
    plt.show()

    env.close()
    print(f"final avg reward (last 50 eps): {np.mean(rewards[-50:]):.1f}")
