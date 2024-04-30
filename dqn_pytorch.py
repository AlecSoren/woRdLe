import gymnasium
from gymnasium.utils.play import play

import wordle_environment

import torch as T
import torch.nn as nn



class DQN(nn.Module):
      
    def __init__(self, env, ):
        super(DQN, self).__init__()

        self.env = env
        
        input_size = gymnasium.spaces.utils.flatten_space(env.observation_space).shape[0]

        input_shape = gymnasium.spaces.utils.flatten_space(env.observation_space).shape
        output_shape = gymnasium.spaces.utils.flatten_space(env.action_space).shape

        self.input_layer = nn.Linear(*input_shape, 10)
        self.hidden_layer_1 = nn.Linear(10, 10)
        self.hidden_layer_2 = nn.Linear(10, 10)
        self.output_layer = nn.Linear(10, *output_shape)

        print(self.input_layer)
        print(self.hidden_layer_1)
        print(self.hidden_layer_2)
        print(self.output_layer)




env = gymnasium.make("CartPole-v1")#, render_mode="human")
env = wordle_environment.make(custom_settings={'action_mode':4})

dqn = DQN(env)


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