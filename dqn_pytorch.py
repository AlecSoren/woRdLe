import gymnasium
import gymnasium.spaces.utils
from gymnasium.utils.play import play

import wordle_environment

import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

import time



class DQN(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_layers, learning_rate):
        super(DQN, self).__init__()

        layer_sizes = [input_size, *hidden_layers, output_size]
        layers = [nn.Linear(s, layer_sizes[i + 1]) for i, s in enumerate(layer_sizes[:-1])]
        stack = [None] * (len(layers) * 2 - 1)
        stack[::2] = layers
        stack[1::2] = [nn.ReLU() for i in range(len(layers) - 1)]
        self.linear_relu_stack = nn.Sequential(*stack)

        self.device = T.device('cpu')
        self.optimiser = optim.Adam(self.parameters(), lr = learning_rate)
        self.loss = nn.MSELoss()
        self.to(self.device)

    def forward(self, nn_input):
        return self.linear_relu_stack(nn_input)
    


class DQN_Agent:

    def __init__(
        self,
        env,
        discount_factor,
        epsilon,
        learning_rate,
        minibatch_size,
        replay_buffer_limit = 100000,
        epsilon_final = 0.01,
        epsilon_decay = 5e-4
    ):
        self.env = env
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.minibatch_size = minibatch_size
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        
        input_size = gymnasium.spaces.utils.flatten_space(env.observation_space).shape[0]
        output_size = gymnasium.spaces.utils.flatten_space(env.action_space).shape[0]
        self.Q_eval = DQN(input_size, output_size, (124, 124, 124), learning_rate)

        self.action_space = env.action_space
        try:
            actions_number = len(env.action_space)
            self.actions_shape = (actions_number, output_size // actions_number)
        except TypeError:
            self.actions_shape = None
        
        replay_buffer_state_shape = (replay_buffer_limit,) + env.observation_space.shape
        replay_buffer_state_dtype = env.observation_space.dtype
        replay_buffer_action_shape = (replay_buffer_limit,) + env.action_space.shape
        replay_buffer_action_dtype = env.action_space.dtype

        self.replay_buffer_states = np.empty(replay_buffer_state_shape, replay_buffer_state_dtype)
        self.replay_buffer_new_states = np.empty(replay_buffer_state_shape, replay_buffer_state_dtype)
        self.replay_buffer_actions = np.empty(replay_buffer_action_shape, replay_buffer_action_dtype)
        self.replay_buffer_rewards = np.empty(replay_buffer_limit, np.float32)
        self.replay_buffer_terminals = np.empty(replay_buffer_limit, np.bool_)

        self.replay_buffer_position = 0
        self.replay_buffer_limit = replay_buffer_limit


    def store_transition(self, state, action, new_state, reward, terminal):
        i = self.replay_buffer_position % self.replay_buffer_limit
        self.replay_buffer_states[i] = state
        self.replay_buffer_new_states[i] = new_state
        self.replay_buffer_actions[i] = action
        self.replay_buffer_rewards[i] = reward
        self.replay_buffer_terminals[i] = terminal
        self.replay_buffer_position += 1


    def choose_action(self, state):
        if np.random.random() > self.epsilon:
            nn_input = T.tensor(state, dtype=T.float32).to(self.Q_eval.device)
            nn_output = self.Q_eval.forward(nn_input).detach().numpy()
            if self.actions_shape:
                actions = nn_output.reshape(self.actions_shape)
                return np.argmax(actions, 1)
            else:
                return np.argmax(nn_output)
        else:
            return self.action_space.sample()
        

    def learn_from_minibatch(self):
        if self.replay_buffer_position < self.minibatch_size:
            return
        
        self.Q_eval.optimiser.zero_grad()

        replay_buffer_size = min(self.replay_buffer_limit, self.replay_buffer_position)
        minibatch_indices = np.random.choice(replay_buffer_size, self.minibatch_size, False)
        
        minibatch_range = np.arange(self.minibatch_size, dtype = np.int32)

        state_minibatch = T.tensor(self.replay_buffer_states[minibatch_indices], dtype=T.float32).to(self.Q_eval.device)
        new_state_minibatch = T.tensor(self.replay_buffer_new_states[minibatch_indices], dtype=T.float32).to(self.Q_eval.device)
        reward_minibatch = T.tensor(self.replay_buffer_rewards[minibatch_indices]).to(self.Q_eval.device)
        terminal_minibatch = T.tensor(self.replay_buffer_terminals[minibatch_indices]).to(self.Q_eval.device)
        action_minibatch = self.replay_buffer_actions[minibatch_indices]

        q_eval = self.Q_eval.forward(state_minibatch)[minibatch_range, action_minibatch]
        q_next = self.Q_eval.forward(new_state_minibatch)
        q_next[terminal_minibatch] = 0.0

        q_target = reward_minibatch + self.discount_factor * T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimiser.step()

        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.epsilon_final \
            else self.epsilon_final
        


env = gymnasium.make("CartPole-v1")#, render_mode="human")
#env = wordle_environment.make(custom_settings={'action_mode':3,'word_length':2,'max_guesses':3,'truncation_limit':10})
agent = DQN_Agent(env, 0.99, 0.9, 0.001, 128)
state, info = env.reset()
total_reward = 0
for i in range(50000):
    action = agent.choose_action(state)
    new_state, reward, terminal, truncated, info = env.step(action)
    env.render()
    total_reward += reward
    agent.store_transition(state, action, new_state, reward, terminal)
    state = new_state
    agent.learn_from_minibatch()
    if terminal or truncated:
        state, info = env.reset()
        print(agent.epsilon, total_reward)
        total_reward = 0


def handwritten_policy(env):
    state, info = env.reset()
    total_reward = 0
    while True:

        angle, angular_velocity = state[2:]
        action = int(angular_velocity > 0)

        state, reward, terminal, truncated, info = env.step(action)
        total_reward += reward
        if terminal or truncated:
                state, info = env.reset()
                print(total_reward)
                total_reward = 0