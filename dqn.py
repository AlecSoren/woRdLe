import numpy as np
from tensorflow import keras
import wordle_environment
import time


# Convert an array of environment states to an array of inputs compatible with neural networks using one hot encoding
# Needed because letters and colours shouldn't be represented numerically
def convert_state_to_input(state):
    letter_encodings = keras.utils.to_categorical(state[0].flatten(), num_classes=27)
    colour_encodings = keras.utils.to_categorical(state[1].flatten(), num_classes=4)
    state_encodings = np.array([np.concatenate([letter_encodings.flatten(), colour_encodings.flatten()])])
    return state_encodings


def create_dqn_nn(input_size):
    nn = keras.models.Sequential([
        keras.Input((input_size,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(26)
    ])
    nn.compile(optimizer='adam', loss=keras.losses.Huber())
    return nn


def dqn(env, replay_buffer_size, num_episodes, epsilon, minibatch_size, discount_factor, network_transfer_freq):
    replay_buffer = np.empty(replay_buffer_size, dtype='object')
    replay_buffer_insert_i = 0  # The position to insert a new value, increment by one each use and wrap to zero
    replay_buffer_samples = 0  # How many samples have been entered into the replay buffer

    init_state = env.reset()[0]
    state_size = convert_state_to_input(init_state).size

    q1 = create_dqn_nn(state_size)  # Action-value network
    q2 = create_dqn_nn(state_size)  # Target action-value network
    q2.set_weights(q1.get_weights())
    network_update_count = 0

    for episode_i in range(num_episodes):
        state = env.reset()[0]
        terminal, truncated = False, False
        total_reward = 0
        while not (terminal or truncated):
            start_time = time.time()
            # Epsilon greedy action choice
            if np.random.random() < epsilon:
                action = np.random.randint(26)
            else:
                action = np.argmax(q1.predict(convert_state_to_input(state), verbose=0))

            # Take step and add experience to replay buffer
            next_state, reward, terminal, truncated, _ = env.step(action)
            total_reward += reward
            replay_buffer[replay_buffer_insert_i] = (state, action, reward, next_state, terminal)
            replay_buffer_insert_i = (replay_buffer_insert_i + 1) % replay_buffer_size
            replay_buffer_samples = min(replay_buffer_samples + 1, replay_buffer_size)

            # Choose mini batch from buffer and use to retrain model
            minibatch_i = np.random.choice(replay_buffer_samples, min(replay_buffer_samples, minibatch_size), replace=False)
            minibatch = replay_buffer[minibatch_i]
            for s_i, (s_state, s_action, s_reward, s_next_state, s_next_terminal) in enumerate(minibatch):
                if s_next_terminal:
                    s_y = s_reward
                else:
                    s_y = s_reward + discount_factor * np.max(q2.predict(convert_state_to_input(s_next_state), verbose=0))
                s_input = convert_state_to_input(s_state)
                s_y_pred = q1.predict(s_input, verbose=0)
                s_y_pred[:, s_action] = s_y
                q1.fit(s_input, s_y_pred, verbose=0)

            # Update target weights of model regularly
            network_update_count += 1
            if network_update_count >= network_transfer_freq:
                q2.set_weights(q1.get_weights())
                network_update_count = 0
            
        print((episode_i, total_reward))


custom_settings = {
    'word_length': 2,
    'truncation_limit': 50
}
custom_render_settings = {'render_mode': 'command_line'}
environment = wordle_environment.make(custom_settings, custom_render_settings)

dqn(environment, replay_buffer_size=1000000, num_episodes=100, epsilon=0.1, minibatch_size=32, discount_factor=0.9, network_transfer_freq=1000)
