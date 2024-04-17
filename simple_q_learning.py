import wordle_environment
import numpy as np
from random import random, randint
from time import time
import json
from datetime import datetime
import base64



def state_to_word(state):
    return ''.join(['abcdefghijklmnopqrstuvwxyz.!'[c] for c in state])



def update_q(q_table, state, action, reward, new_state, step_size, discount_factor):
    current_q = q_table[state][action]
    future_q = max(q_table[new_state])
    td_error = reward + discount_factor * future_q - current_q
    q_table[state][action] += step_size * td_error



def q_learning(env, episode_limit, step_size, epsilon, discount_factor, initial_q, q_table = None):
    if q_table == None:
        q_table = {env.reset()[0].tobytes():[initial_q]*26}

    rewards_per_episode = []

    completion_time = time()

    for episode in range(episode_limit):

        if episode != 0 and episode % (episode_limit // 100) == 0:
            print(f'{int((episode / episode_limit) * 100)}% done, {round(time() - completion_time, 1)} s')

        state = env.reset()[0].tobytes()
        terminal = False
        total_reward = 0

        while True:
            if random() > epsilon:
                q_values = q_table[state]
                action = q_values.index(max(q_values))
            else:
                action = randint(0, 25)
            new_state, reward, terminal, truncated, info = env.step(action)
            new_state = new_state.tobytes()

            if new_state not in q_table:
                q_table[new_state] = [initial_q]*26

            update_q(q_table, state, action, reward, new_state, step_size, discount_factor)

            state = new_state
            total_reward += reward

            if terminal or truncated:
                break

        rewards_per_episode.append(total_reward)

    return q_table, rewards_per_episode



custom_settings = {'word_length':5,'truncation_limit':1000}
episodes = 1000000
env = wordle_environment.make(custom_settings)

try:
    with open('q_table.json') as f:
        q_table = json.load(f)
        q_table = {base64.b64decode(k):q_table[k] for k in q_table}
except FileNotFoundError:
    q_table = {env.reset()[0].tobytes():[0]*26}
    print('Could not load Q-table, initialising empty')
else:
    print('Loaded Q-table from file')

q_table, rewards_per_episode = q_learning(env, episodes, 0.5, 0.001, 0.5, 0, q_table)

file_path = 'q_table_archive/q_table_' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '.json'
encoded_q_table = {base64.b64encode(k).decode('utf-8'):q_table[k] for k in q_table}
with open(file_path, 'w') as f:
    json.dump(encoded_q_table, f)

print('Initial average reward: ' + str(np.average(rewards_per_episode[:1000])))
print('Final average reward: ' + str(np.average(rewards_per_episode[1000:])))