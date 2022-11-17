import numpy as np
import gym

class QAgent:
    def __init__(self, env, discount_factor, learning_rate, epsilon, decay_rate):
        self.gamma = discount_factor
        self.lr = learning_rate
        self.eps = epsilon
        self.dr = decay_rate
        self.terminal_states = self.get_terminal(env=env)
        print('State Space: \n', env.desc.astype(str))
        print('Terminal States Are At: \n', self.terminal_states)

        action_size = env.action_space.n
        state_size = env.observation_space.n
        self.Q_values = np.zeros((state_size, action_size))

    def get_terminal(self, env):
        states_as_str = env.desc.astype('str')
        one_dim_states = np.reshape(states_as_str, (1, states_as_str.shape[0]*states_as_str.shape[1]))
        terminals = np.where((one_dim_states == 'H') | (one_dim_states == 'G'))[1]
        return terminals.tolist()

    def action(self, s):
        sample_proba = np.random.uniform()
        # print(sample_proba)
        if sample_proba > self.eps:
            action = np.argmax(self.Q_values[s, :])
        else: 
            action = env.action_space.sample()
        
        # self.eps *= 1
        return action

    def update_Q(self, s, a, r, s_tag):
        self.Q_values[s, a] = self.Q_values[s, a] + self.lr * (r + self.gamma * np.max(self.Q_values[s_tag, :]) - self.Q_values[s, a])
        # print(self.Q_values[s, a])

env = gym.make('FrozenLake-v1', new_step_api=True, max_episode_steps=100, map_name='4x4')
# env = gym.make('FrozenLake-v1', max_episode_steps=100, map_name='4x4')

##### Hyperparameters:
epsilon = 0.3
decay_rate = 0.99
learning_rate = 0.1
gamma = 0.95

agent = QAgent(env=env, discount_factor=gamma, learning_rate=learning_rate, epsilon=epsilon, decay_rate=decay_rate)

num_episodes = 15000
rewards = np.zeros((num_episodes))
for i in range(num_episodes):
    tot_rewards = 0
    s = env.reset()
    while True:
        a = agent.action(s)
        s_tag, r, done, _, _ = env.step(a)
        agent.update_Q(s, a, r, s_tag)

        s = s_tag
        tot_rewards += r

        if done:
            # print('done')
            # print(tot_rewards)
            break

    rewards[i] = tot_rewards

print('Final Q Table:')
print(agent.Q_values)

