import numpy as np
from tensorflow import keras
import wordle_environment


# Convert an array of environment states to an array of inputs compatible with neural networks using one hot encoding
# Needed because letters and colours shouldn't be represented numerically
def convert_state_to_input(state):
    letter_encodings = keras.utils.to_categorical(state[0].flatten(), num_classes=27)
    colour_encodings = keras.utils.to_categorical(state[1].flatten(), num_classes=4)
    state_encodings = np.concatenate([letter_encodings.flatten(), colour_encodings.flatten()])
    return state_encodings


def create_dqn_nn(input_size):
    nn = keras.models.Sequential([
        keras.Input((input_size,)),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(26)
    ])
    nn.compile(optimizer='adam', loss=keras.losses.Huber())
    return nn


def dqn(env, replay_buffer_size, num_episodes, epsilon, minibatch_size, discount_factor, network_transfer_freq, load_weights_path=None, save_weights_path=None):
    init_state = env.reset()[0]
    input_init_state = convert_state_to_input(init_state)

    replay_buffer_states = np.empty((replay_buffer_size,) + input_init_state.shape)
    replay_buffer_actions = np.empty(replay_buffer_size, dtype=int)
    replay_buffer_rewards = np.empty(replay_buffer_size)
    replay_buffer_next_states = np.empty((replay_buffer_size,) + input_init_state.shape)
    replay_buffer_terminal = np.empty(replay_buffer_size, dtype=bool)
    replay_buffer_insert_i = 0  # The position to insert a new value, increment by one each use and wrap to zero
    replay_buffer_samples = 0  # How many samples have been entered into the replay buffer

    q1 = create_dqn_nn(input_init_state.size)  # Action-value network
    q2 = create_dqn_nn(input_init_state.size)  # Target action-value network
    if load_weights_path:
        q1.load_weights(load_weights_path)
    q2.set_weights(q1.get_weights())
    network_update_count = 0

    for episode_i in range(num_episodes):
        print('Episode', episode_i + 1, '/', num_episodes)
        state = env.reset()[0]
        terminal, truncated = False, False
        while not (terminal or truncated):
            # Epsilon greedy action choice
            if np.random.random() < epsilon:
                action = np.random.randint(26)
            else:
                action = np.argmax(q1.predict(np.array([convert_state_to_input(state)]), verbose=0))

            # Take step and add experience to replay buffer
            next_state, reward, terminal, truncated, _ = env.step(action)
            replay_buffer_states[replay_buffer_insert_i] = convert_state_to_input(state)
            replay_buffer_actions[replay_buffer_insert_i] = action
            replay_buffer_rewards[replay_buffer_insert_i] = reward
            replay_buffer_next_states[replay_buffer_insert_i] = convert_state_to_input(next_state)
            replay_buffer_terminal[replay_buffer_insert_i] = terminal
            replay_buffer_insert_i = (replay_buffer_insert_i + 1) % replay_buffer_size
            replay_buffer_samples = min(replay_buffer_samples + 1, replay_buffer_size)

            # Choose mini batch from buffer for model training
            minibatch_i = np.random.choice(replay_buffer_samples, min(replay_buffer_samples, minibatch_size), replace=False)
            minibatch_states = replay_buffer_states[minibatch_i]
            minibatch_actions = replay_buffer_actions[minibatch_i]
            minibatch_rewards = replay_buffer_rewards[minibatch_i]
            minibatch_next_states = replay_buffer_next_states[minibatch_i]
            minibatch_terminal = replay_buffer_terminal[minibatch_i]

            # Q-values that the mini batch state-actions map too
            minibatch_y = np.empty(minibatch_i.size)
            minibatch_y[minibatch_terminal] = minibatch_rewards[minibatch_terminal]
            minibatch_y[~minibatch_terminal] = minibatch_rewards[~minibatch_terminal] + discount_factor * np.max(q2.predict(minibatch_next_states[~minibatch_terminal], verbose=0), axis=1)

            # The network's predictions for other actions are needed for training
            minibatch_y_preds = q1.predict(minibatch_states, verbose=0)
            minibatch_y_preds[:, minibatch_actions] = minibatch_y
            q1.fit(minibatch_states, minibatch_y_preds, verbose=0)

            # Update target weights of model regularly
            network_update_count += 1
            if network_update_count >= network_transfer_freq:
                q2.set_weights(q1.get_weights())
                network_update_count = 0

            state = next_state

        # Save after each episode
        if save_weights_path:
            q1.save_weights(save_weights_path)



if __name__ == '__main__':
    custom_settings = {
        'word_length': 2,
        'truncation_limit': 1000,
        'invalid_word_reward': -0.01
    }
    custom_render_settings = {'render_mode': 'command_line', 'animation_duration': 1e-8}
    environment = wordle_environment.make(custom_settings, custom_render_settings)
    weights_path = 'network.weights.h5'

    dqn(environment, replay_buffer_size=1000000, num_episodes=1, epsilon=0.25, minibatch_size=32, discount_factor=0.9, network_transfer_freq=1000, load_weights_path=None, save_weights_path=weights_path)
