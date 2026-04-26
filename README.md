RL algorithms from scratch

implementing rl algorithms from sutton and barto as i go through the book. started this to actually understand whats happening under the hood instead of just calling stable-baselines.

algorithms so far

bandits
- epsilon greedy (ch. 2)

dynamic programming
- value iteration on FrozenLake (ch. 4)

TD learning
- SARSA (ch. 6)
- Q-Learning (ch. 6)
- comparison of both on CliffWalking

deep RL
- DQN on CartPole (replay buffer + target network)

how to run

pip install -r requirements.txt

then just run whichever script you want:

python bandits/epsilon_greedy.py
python dynamic_programming/value_iteration.py
python td_learning/q_learning.py
python td_learning/sarsa.py
python td_learning/compare.py
python deep_rl/dqn.py

each script saves a plot to its own folder.

notes

still working through the book so more algorithms coming. planning to add monte carlo methods and policy gradients next.
