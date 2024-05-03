import sys
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd

import wordle_environment

#######################################################################
# COPIED FROM INTERNET TO EXPERIMENT WITH CURIOSITY - NOT SUBMITTABLE #
#######################################################################

# hyperparameters

hidden_size = 512
learning_rate = 3e-4

# Constants
GAMMA = 0.99
num_steps = 1000000
max_episodes = 1000000


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4):
        super(ActorCritic, self).__init__()

        self.num_actions = num_actions
        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, hidden_size)
        self.critic_linear3 = nn.Linear(hidden_size, hidden_size)
        self.critic_linear4 = nn.Linear(hidden_size, hidden_size)
        self.critic_linear5 = nn.Linear(hidden_size, 1)

        self.actor_linear1 = nn.Linear(num_inputs, hidden_size)
        self.actor_linear2 = nn.Linear(hidden_size, hidden_size)
        self.actor_linear3 = nn.Linear(hidden_size, hidden_size)
        self.actor_linear4 = nn.Linear(hidden_size, hidden_size)
        self.actor_linear5 = nn.Linear(hidden_size, num_actions)

        self.to(torch.device('cuda'))

    def forward(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0)).to(torch.device('cuda'))

        value = F.relu(self.critic_linear1(state))
        value = F.relu(self.critic_linear2(value))
        value = F.relu(self.critic_linear3(value))
        value = F.relu(self.critic_linear4(value))
        value = self.critic_linear5(value)

        policy_dist = F.relu(self.actor_linear1(state))
        policy_dist = F.relu(self.actor_linear2(policy_dist))
        policy_dist = F.relu(self.actor_linear3(policy_dist))
        policy_dist = F.relu(self.actor_linear4(policy_dist))
        policy_dist = F.softmax(self.actor_linear5(policy_dist), dim=1)

        return value.cpu(), policy_dist.cpu()

    def save_weights(self, file_path):
        torch.save(self.state_dict(), file_path)


def a2c(env):
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.n

    actor_critic = ActorCritic(num_inputs, num_outputs, hidden_size)
    ac_optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)

    all_lengths = []
    average_lengths = []
    all_rewards = []
    entropy_term = 0

    for episode in range(max_episodes):
        if episode % 1000 == 0:
            actor_critic.save_weights('weights.pt')
        log_probs = []
        values = []
        rewards = []

        state, _ = env.reset()
        for steps in range(num_steps):
            value, policy_dist = actor_critic.forward(state)
            value = value.detach().numpy()[0, 0]
            dist = policy_dist.detach().numpy()

            action = np.random.choice(num_outputs, p=np.squeeze(dist))
            log_prob = torch.log(policy_dist.squeeze(0)[action])
            entropy = -np.sum(np.mean(dist) * np.log(dist))
            new_state, reward, done, _, _ = env.step(action)
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            entropy_term += entropy
            state = new_state

            if done or steps == num_steps - 1:
                Qval, _ = actor_critic.forward(new_state)
                Qval = Qval.detach().numpy()[0, 0]
                all_rewards.append(np.sum(rewards))
                all_lengths.append(steps)
                average_lengths.append(np.mean(all_lengths[-10:]))
                if episode % 10 == 0:
                    sys.stdout.write("episode: {}, reward: {}, total length: {}, average length: {} \n".format(episode,
                                                                                                               np.sum(
                                                                                                                   rewards),
                                                                                                               steps,
                                                                                                               average_lengths[
                                                                                                                   -1]))
                break

        # compute Q values
        Qvals = np.zeros_like(values)
        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + GAMMA * Qval
            Qvals[t] = Qval

        # update actor critic
        values = torch.FloatTensor(values)
        Qvals = torch.FloatTensor(Qvals)
        log_probs = torch.stack(log_probs)

        advantage = Qvals - values
        actor_loss = (-log_probs * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss + 0.001 * entropy_term

        ac_optimizer.zero_grad()
        ac_loss.backward()
        ac_optimizer.step()

    # Plot results
    smoothed_rewards = pd.Series.rolling(pd.Series(all_rewards), 10).mean()
    smoothed_rewards = [elem for elem in smoothed_rewards]
    plt.plot(all_rewards)
    plt.plot(smoothed_rewards)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

    plt.plot(all_lengths)
    plt.plot(average_lengths)
    plt.xlabel('Episode')
    plt.ylabel('Episode length')
    plt.show()


if __name__ == '__main__':
    custom_settings = {
        'word_length': 3,
        'truncation_limit': 1000000,
        # 'max_hidden_word_options': 50,
        # 'hidden_word_subset_seed': 1,
        'correct_guess_reward': 1,
        'early_guess_reward': 0.2,
        'colour_rewards': (0, 0.05, 0.1),
        'invalid_word_reward': 0,
        'step_reward': -0.0001,
        'repeated_guess_reward': 0
    }
    custom_render_settings = {'render_mode': 'gui', 'animation_duration': 0}
    environment = wordle_environment.make(custom_settings, custom_render_settings)

    a2c(environment)
