import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from q_learning import train as q_train
from sarsa import train as sarsa_train

# SARSA vs Q-learning on CliffWalking
# they converge to different policies which is the interesting part
# Q-learning finds the optimal (risky) path along the cliff edge
# SARSA stays further from the cliff because it accounts for its own exploration

env = gym.make("CliffWalking-v0")

runs = 10
episodes = 300
q_all = np.zeros((runs, episodes))
sarsa_all = np.zeros((runs, episodes))

print("running comparison...")
for i in range(runs):
    _, q_rewards = q_train(env, episodes=episodes)
    _, sarsa_rewards = sarsa_train(env, episodes=episodes)
    q_all[i] = q_rewards
    sarsa_all[i] = sarsa_rewards
    print(f"  run {i+1}/{runs} done")

q_mean = q_all.mean(axis=0)
sarsa_mean = sarsa_all.mean(axis=0)

def smooth(x, w=20):
    return np.convolve(x, np.ones(w)/w, mode="valid")

plt.figure(figsize=(10, 5))
plt.plot(smooth(q_mean), label="Q-Learning", color="blue")
plt.plot(smooth(sarsa_mean), label="SARSA", color="orange")
plt.xlabel("episode")
plt.ylabel("avg reward")
plt.title("SARSA vs Q-Learning on CliffWalking (avg over 10 runs)")
plt.legend()
plt.tight_layout()
plt.savefig("td_learning/comparison.png", dpi=120)
plt.show()

env.close()
print("\nQ-Learning final avg:", round(q_mean[-50:].mean(), 1))
print("SARSA final avg:", round(sarsa_mean[-50:].mean(), 1))
