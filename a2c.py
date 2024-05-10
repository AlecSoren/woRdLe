import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np
import os

import wordle_environment

DEVICE = T.device('cpu')


# Generates a CNN with specified layers, with ReLU activation in between and optional softmax output
# Contains 1 convolutional layer with pointwise filters to extract features from each grid cell
class CNN(nn.Module):

    def __init__(self, input_shape, output_size, conv_filters, hidden_layers, learning_rate, softmax=False):
        super(CNN, self).__init__()
        layers = [nn.Conv2d(input_shape[2], conv_filters, 1), nn.Flatten(0, -1)]
        layer_sizes = [input_shape[0] * input_shape[1] * conv_filters, *hidden_layers, output_size]
        for i, s in enumerate(layer_sizes[:-1]):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(s, layer_sizes[i + 1]))
        if softmax:
            layers.append(nn.Softmax())
        self.sequential = nn.Sequential(*layers)
        self.optimiser = optim.Adam(self.parameters(), lr=learning_rate)
        self.to(DEVICE)

    def forward(self, nn_input):
        nn_input = nn_input.permute(2, 0, 1)
        return self.sequential(nn_input)

    def save_weights(self, file_path):
        T.save(self.state_dict(), file_path)

    def load_weights(self, file_path):
        self.load_state_dict(T.load(file_path, map_location = DEVICE))


# Actor network to generate probability distribution of actions given a state
class Actor(CNN):

    def __init__(self, input_size, output_size, hidden_layers, learning_rate):
        super().__init__(input_size, output_size, 64, hidden_layers, learning_rate, softmax=True)


# Critic network to valuate states
class Critic(CNN):

    def __init__(self, input_size, hidden_layers, learning_rate):
        super().__init__(input_size, 1, 64, hidden_layers, learning_rate)


# RND-based computational curiosity
# Target network is a constant, randomly initialised network generating mappings between states and a vector
# Predictor network attempts to replicate the target network and trains on its outputs
# The predictor will produce lower errors from states similar to those it has previously seen
class FeatureTargetPredictor:

    def __init__(self, input_size, output_size, conv_filters, hidden_layers, learning_rate):
        self.feature_target = CNN(input_size, output_size, conv_filters, hidden_layers, learning_rate)
        self.feature_predictor = CNN(input_size, output_size, conv_filters, hidden_layers, learning_rate)

    # Compare predictions of target and predictor networks and return prediction error
    # Perform backpropagation on given sample
    def update(self, state):
        target_features = self.feature_target.forward(state)
        predicted_features = self.feature_predictor.forward(state)

        self.feature_predictor.optimiser.zero_grad()
        loss = nn.MSELoss()(target_features, predicted_features)
        loss.backward()
        self.feature_predictor.optimiser.step()

        prediction_error = np.sum((target_features.cpu().detach().numpy() - predicted_features.cpu().detach().numpy()) ** 2)
        return prediction_error

    def save_weights(self, file_path_target, file_path_predictor):
        T.save(self.feature_target.state_dict(), file_path_target)
        T.save(self.feature_predictor.state_dict(), file_path_predictor)

    def load_weights(self, file_path_target, file_path_predictor):
        self.feature_target.load_state_dict(T.load(file_path_target, map_location = DEVICE))
        self.feature_predictor.load_state_dict(T.load(file_path_predictor, map_location = DEVICE))


def a2c(
        env,
        num_episodes,
        save_weight_dir=None,
        load_weight_dir=None,
        explore_reward_scalar=1.0,
        feature_size=64,
        discount_factor=0.9,
        learning_rate=1e-4,
        hidden_layers=(512, 256, 128),
        conv_layers=64,
        entropy_scalar=0.01,
        learn=True,
        show_boards = False
):
    input_shape = env.observation_space.shape
    output_size = env.action_space.n

    actor = Actor(input_shape, output_size, hidden_layers, learning_rate)
    critic = Critic(input_shape, hidden_layers, learning_rate)

    feature_target_predictor = FeatureTargetPredictor(input_shape, feature_size, conv_layers, (128, 128, 128), learning_rate)

    hidden_word_i = 0

    episode_rewards = []
    episode_explore_rewards = []
    if load_weight_dir:
        actor.load_weights(f'{load_weight_dir}/actor_weights.pt')
        critic.load_weights(f'{load_weight_dir}/critic_weights.pt')
        feature_target_predictor.load_weights(f'{load_weight_dir}/feature_target_weights.pt',
                                              f'{load_weight_dir}/feature_predictor_weights.pt')
        episode_rewards = np.load(f'{load_weight_dir}/episode_rewards.npy').tolist()
        episode_explore_rewards = np.load(f'{load_weight_dir}/episode_explore_rewards.npy').tolist()

    while len(episode_rewards) < num_episodes:
        episode_i = len(episode_rewards) + 1
        if save_weight_dir and learn:
            if episode_i % 25 == 0:
                actor.save_weights(f'{save_weight_dir}/actor_weights.pt')
                critic.save_weights(f'{save_weight_dir}/critic_weights.pt')
                feature_target_predictor.save_weights(f'{save_weight_dir}/feature_target_weights.pt',
                                                      f'{save_weight_dir}/feature_predictor_weights.pt')
                np.save(f'{save_weight_dir}/episode_rewards.npy', np.array(episode_rewards))
                np.save(f'{save_weight_dir}/episode_explore_rewards.npy', np.array(episode_explore_rewards))

            if episode_i in [2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000, 25000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]:
                episode_dir = f'{save_weight_dir}/{episode_i}'
                if not os.path.exists(episode_dir):
                    os.mkdir(episode_dir)
                actor.save_weights(f'{episode_dir}/actor_weights.pt')
                critic.save_weights(f'{episode_dir}/critic_weights.pt')
                feature_target_predictor.save_weights(f'{episode_dir}/feature_target_weights.pt',
                                                      f'{episode_dir}/feature_predictor_weights.pt')

        if show_boards:
            #hidden_words = [e[0] for e in env.unwrapped.hidden_words]
            hidden_words = ('ace', 'dad', 'gah')
            state, _ = env.reset(hidden_words[hidden_word_i % len(hidden_words)])
            hidden_word_i += 1
        else:
            state, _ = env.reset()
        values = []
        rewards = []
        total_explore_rewards = 0
        log_probs = []
        entropy = 0
        terminal = False
        truncated = False

        while not terminal and not truncated:
            # Get action distribution and state value from actor and critic
            state_input = T.FloatTensor(state).to(DEVICE)
            action_dist = Categorical(actor.forward(state_input))
            value = critic.forward(state_input)

            # Choose action based on distribution and step
            action = action_dist.sample()
            next_state, reward, terminal, truncated, _ = env.step(action.cpu().numpy())

            # Add exploration reward for seeing unfamiliar states
            next_state_input = T.FloatTensor(next_state).to(DEVICE)
            explore_reward = explore_reward_scalar * feature_target_predictor.update(next_state_input)
            reward += explore_reward
            total_explore_rewards += explore_reward

            values.append(value)
            rewards.append(reward)
            log_probs.append(action_dist.log_prob(action))
            entropy += action_dist.entropy().mean()

            state = next_state

        if show_boards:
            env.unwrapped.render()
            input()

        print('Episode: {:8}  |  Total Reward: {:16.8f}  |  Total Explore Reward: {:16.8f}'.format(episode_i, sum(rewards), total_explore_rewards))
        episode_rewards.append(sum(rewards) - total_explore_rewards)
        episode_explore_rewards.append(total_explore_rewards)

        if not learn:
            continue

        # Calculate returns for episode (working backwards)
        next_value = 0
        if not terminal:
            next_value = critic.forward(T.FloatTensor(state).to(DEVICE))
        returns = []
        for i in reversed(range(len(values))):
            next_value = rewards[i] + discount_factor * next_value
            returns.insert(0, next_value)

        values = T.cat(values)
        log_probs = T.stack(log_probs).to(DEVICE)
        returns = T.tensor(returns).to(DEVICE)

        # Calculate advantages as difference between returns and critic values
        advantages = returns - values

        # Calculate losses based on log action probabilities and advantages
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = 0.5 * advantages.pow(2).mean()
        loss = actor_loss + critic_loss - entropy_scalar * entropy

        # Train networks
        actor.optimiser.zero_grad()
        critic.optimiser.zero_grad()
        loss.backward()
        actor.optimiser.step()
        critic.optimiser.step()


if __name__ == '__main__':
    custom_settings = {
        'word_length': 3,
        'truncation_limit': 1000,
        'correct_guess_reward': 10,
        'early_guess_reward': 0.2,
        'colour_rewards': (0, 0.05, 0.1),
        'valid_word_reward': 0,
        'invalid_word_reward': 0,
        'step_reward': -0.0001,
        'repeated_guess_reward': 0,
        'alphabet': 'abcdefgh',
        'vocab_file': 'word_lists/three_letter_abcdefgh.txt',
        'hidden_words_file': 'word_lists/three_letter_abcdefgh.txt',
        'state_representation': 'one_hot_grid'
    }
    #custom_render_settings = {'render_mode': 'gui', 'animation_duration': 0.5}
    custom_render_settings = {'render_mode': 'gui', 'animation_duration': 0}
    environment = wordle_environment.make(custom_settings, custom_render_settings)

    a2c(
        environment,
        80030,
        save_weight_dir=None,
        load_weight_dir='final_model',
        explore_reward_scalar=1,
        feature_size=128,
        discount_factor=0.9,
        learning_rate=5e-4,
        hidden_layers=(256, 128),
        conv_layers=64,
        entropy_scalar=0.001,
        learn=False,
        show_boards=True
    )
