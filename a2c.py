import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np

import wordle_environment

DEVICE = T.device('cpu')


# Generates a NN with specified layers, with ReLU activation in between and optional softmax output
class NN(nn.Module):

    def __init__(self, input_size, output_size, hidden_layers, learning_rate, softmax=False):
        super(NN, self).__init__()

        layer_sizes = [input_size, *hidden_layers, output_size]
        layers = [nn.Linear(s, layer_sizes[i + 1]) for i, s in enumerate(layer_sizes[:-1])]
        stack = [None] * (len(layers) * 2 - 1)
        stack[::2] = layers
        stack[1::2] = [nn.ReLU() for _ in range(len(layers) - 1)]
        if softmax:
            stack.append(nn.Softmax())
        self.linear_relu_stack = nn.Sequential(*stack)
        self.optimiser = optim.Adam(self.parameters(), lr=learning_rate)
        self.to(DEVICE)

    def forward(self, nn_input):
        return self.linear_relu_stack(nn_input)

    def save_weights(self, file_path):
        T.save(self.state_dict(), file_path)

    def load_weights(self, file_path):
        self.load_state_dict(T.load(file_path))


# Actor network to generate probability distribution of actions given a state
class Actor(NN):

    def __init__(self, input_size, output_size, hidden_layers, learning_rate):
        super().__init__(input_size, output_size, hidden_layers, learning_rate, softmax=True)


# Critic network to valuate states
class Critic(NN):

    def __init__(self, input_size, hidden_layers, learning_rate):
        super().__init__(input_size, 1, hidden_layers, learning_rate)


# RND-based computational curiosity
# Target network is a constant, randomly initialised network generating mappings between states and a vector
# Predictor network attempts to replicate the target network and trains on its outputs
# The predictor will produce lower errors from states similar to those it has previously seen
class FeatureTargetPredictor:

    def __init__(self, input_size, output_size, hidden_layers, learning_rate):
        self.feature_target = NN(input_size, output_size, hidden_layers, learning_rate)
        self.feature_predictor = NN(input_size, output_size, hidden_layers, learning_rate)

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
        self.feature_target.load_state_dict(T.load(file_path_target))
        self.feature_predictor.load_state_dict(T.load(file_path_predictor))


def a2c(
        env,
        num_episodes,
        save_weight_filepaths=None,
        load_weight_filepaths=None,
        explore_reward_scalar=1.0,
        feature_size=64,
        discount_factor=0.9,
        learning_rate=1e-4,
        hidden_layers=(512, 256, 128)
):
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n

    actor = Actor(input_size, output_size, hidden_layers, learning_rate)
    critic = Critic(input_size, hidden_layers, learning_rate)

    feature_target_predictor = FeatureTargetPredictor(input_size, feature_size, (128, 128, 128), learning_rate)

    if load_weight_filepaths:
        actor.load_weights(save_weight_filepaths[0])
        critic.load_weights(save_weight_filepaths[1])
        feature_target_predictor.load_weights(save_weight_filepaths[2], save_weight_filepaths[3])

    for episode_i in range(num_episodes):
        if episode_i % 50 == 0 and save_weight_filepaths:
            actor.save_weights(save_weight_filepaths[0])
            critic.save_weights(save_weight_filepaths[1])
            feature_target_predictor.save_weights(save_weight_filepaths[2], save_weight_filepaths[3])

        state, _ = env.reset()
        values = []
        rewards = []
        total_explore_rewards = 0
        log_probs = []
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

            state = next_state

        print('Episode: {:8}  |  Total Reward: {:16.8f}  |  Total Explore Reward: {:16.8f}'.format(episode_i + 1, sum(rewards), total_explore_rewards))

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
        loss = actor_loss + critic_loss

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
        'correct_guess_reward': 2,
        'early_guess_reward': 0.2,
        'colour_rewards': (0, 0.05, 0.1),
        'valid_word_reward': 0,
        'invalid_word_reward': 0,
        'step_reward': -0.0001,
        'repeated_guess_reward': 0,
        'alphabet': 'abcd',
        'vocab_file': 'word_lists/three_letter_abcd_all.txt',
        'hidden_words_file': 'word_lists/three_letter_abcd_all.txt',
    }
    custom_render_settings = {'render_mode': 'gui', 'animation_duration': 0}
    environment = wordle_environment.make(custom_settings, custom_render_settings)
    weight_filepaths = ('actor_w_a4.pt', 'critic_w_a4.pt',
                        'f_target_w_a4.pt', 'f_prediction_w_a4.pt')

    a2c(
        environment,
        1000000,
        save_weight_filepaths=weight_filepaths,
        load_weight_filepaths=None,
        explore_reward_scalar=1,
        feature_size=128,
        discount_factor=0.9,
        learning_rate=5e-4,
        hidden_layers=(512, 256, 128)
    )
