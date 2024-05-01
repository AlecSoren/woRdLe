import gymnasium
import wordle_environment

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np



class PPO_NN(nn.Module):

    def __init__(self, input_size, output_size, hidden_layers, learning_rate):
        super(PPO_NN, self).__init__()

        #Create shared layers for actor and critic networks
        layer_sizes = [input_size, *hidden_layers]
        shared_layers = [None] * len(layer_sizes[:-1]) * 2
        shared_layers[::2] = [nn.Linear(s, layer_sizes[i + 1]) for i, s in enumerate(layer_sizes[:-1])]
        shared_layers[1::2] = [nn.ReLU() for i in range(len(layer_sizes[:-1]))]
        
        self.actor = nn.Sequential(*shared_layers, nn.Linear(layer_sizes[-1], output_size))
        self.critic = nn.Sequential(*shared_layers, nn.Linear(layer_sizes[-1], 1))

        self.device = T.device('cpu')
        self.optimiser = optim.Adam(self.parameters(), lr = learning_rate, eps=1e-5)
        self.loss = nn.MSELoss()
        self.to(self.device)


    def forward(self, nn_input):
        probabilities = F.softmax(self.actor(nn_input))
        prediction = self.critic(nn_input)
        return probabilities, prediction
    


class PPO_Agent:

    def __init__(self, env, batch_size):
        self.env = env
        self.batch_size = batch_size

        self.state = env.reset()[0]


    def __collect_trajectories(self):
        batch_states, batch_actions, batch_rewards = [], [], []

        state = self.state
        for s in range(self.batch_size):
            action = self.choose_action(state)

            batch_states.append(state)
            batch_actions.append(action)
            state, reward, terminal, truncated, info = self.env.step(action)
            batch_rewards.append(reward)

            if terminal or truncated:
                state = self.env.reset()[0]

        return batch_states, batch_actions, batch_rewards


    def choose_action(self, state):
        pass


    def train(self, epochs):
        self.state = self.env.reset()[0]

        for e in range(epochs):
            batch_states, batch_actions, batch_rewards = self.__collect_trajectories()



network = PPO_NN(12, 4, (16, 8, 4), 1)
print(network.actor)
print(network.critic)