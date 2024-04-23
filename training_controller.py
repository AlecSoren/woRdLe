import numpy as np
from tensorflow import keras
from time import time

from wordle_environment import make
from dqn import *



def evaluate_network(env, nn):

    time_limit = 600

    episodes = 0
    total_reward = 0
    wrong_guess_num = 0
    games_won = 0

    start_time = time()

    while time() - start_time < time_limit:
        state = convert_state_to_input(env.reset()[0])
        terminal, truncated = False, False
        while not (terminal or truncated):
            action = np.argmax(nn.predict(state, verbose=0))
            #action = 1
            raw_state, reward, terminal, truncated, info = env.step(action)
            state = convert_state_to_input(raw_state)

            total_reward += reward
            wrong_guess_num += info['invalid_word']

        episodes += 1
        games_won += reward >= 1

    return (episodes, total_reward / episodes, wrong_guess_num / episodes, games_won)



def train_network(env, nn):
    try:
        while True:
            dqn(
                env,
                nn,
                replay_buffer_size=1000000,
                num_episodes=1000000,
                epsilon=0.1,
                minibatch_size=32,
                discount_factor=0.9,
                network_transfer_freq=1000
            )
    except KeyboardInterrupt:
        nn.save('network.keras')



def load_model(path = 'network.keras'):
    return keras.models.load_model(path)



env = make(custom_settings = {'word_length':2})
nn = create_dqn_nn(env)
train_network(env, nn)
