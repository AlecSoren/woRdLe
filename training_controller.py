import numpy as np
from tensorflow import keras
from time import time

from wordle_environment import make
from dqn import *



def evaluate_network(env, nn, time_limit = 600):

    episodes = 0
    total_reward = 0
    wrong_guess_num = 0
    games_won = 0

    start_time = time()

    while time() - start_time < time_limit:
        state = convert_state_to_input(env.reset()[0])
        terminal, truncated = False, False
        while not (terminal or truncated):
            action = np.argmax(nn.predict(np.array([state]), verbose=0))
            raw_state, reward, terminal, truncated, info = env.step(action)
            state = convert_state_to_input(raw_state)

            total_reward += reward
            wrong_guess_num += info['invalid_word']

        episodes += 1
        games_won += reward >= 1

    return (episodes, total_reward / episodes, wrong_guess_num / episodes, games_won)



def train_network(env):
    while True:
        try:
            dqn(
                env,
                replay_buffer_size=1000000,
                num_episodes=100000000,
                epsilon=0.25,
                minibatch_size=32,
                discount_factor=0.9,
                network_transfer_freq=1000,
                load_weights_path='network.weights.h5',
                save_weights_path='network.weights.h5'
            )
        except KeyboardInterrupt:
            return
        except OSError:
            pass



def load_network(env, weights_path = 'network.weights.h5'):
    init_state = env.reset()[0]
    input_init_state = convert_state_to_input(init_state)
    q1 = create_dqn_nn(input_init_state.size)
    q1.load_weights(weights_path)
    return q1



env = make(custom_settings = {
    'word_length':2,
    'truncation_limit':24,
    'final_guess_rewards':(0,0.1,0.2),
    'invalid_word_reward':-0.01
})
train_network(env)

nn = load_network(env)
print(evaluate_network(env, nn, 60))