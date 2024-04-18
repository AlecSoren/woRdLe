# woRdLe
Solving Wordle with reinforcement learning. For University of Bath unit CM50270

Requires NumPy and pygame

## Getting started:

import wordle_environment

env = wordle_environment.make()

env.play()

## make()

Returns a Wordle environment instance.

Optionally, accepts a custom_settings dictionary with any of the following fields:
- 'word_length' - 5 by default. If not 2-5, you must specify a 'vocab_file' and 'hidden_words_file'.
- 'alphabet' - Valid characters the player can input. 'abcdefghijklmnopqrstuvwxyz' by default.
- 'vocab_file' - Path to a text file containing the valid word list.
- 'hidden_words_file' - Path to a text file containing all possible hidden/answer words.
- 'max_hidden_word_options' - If specified, plays with a pseudorandom subset of the hidden word list.
Gradually increase the subset size to make it easier for the agent at the start.
- 'hidden_word_subset_seed' - Determines the order in which words are chosen for the subset.
Make sure to use the same seed each time, or it will choose a new random subset of words.
- 'action_mode' - How much control over input is given to the agent. 3 by default.
    - 1 = Full control over all keys. Backspace is -1, enter is -2
    - 2 = Controls letter keys and backspace (-1), but automatically presses enter after filling row.
    - 3 = Controls letter keys only. Automatically enters after five letters and clears row if invalid.
    - 4 = Inputs one row at a time. Each action must be an iterable.
    - 5 = Chooses word from vocab list.
- 'max_guesses' - The number of rows on the Wordle board. 6 by default.
- 'correct_guess_rewards' - Iterable containing rewards for a correct guess after i incorrect guesses,
where i is the index of the iterable. (7, 6, 5, 4, 3, 2) by default.
- 'final_guess_rewards' - Iterable containing rewards for grey, yellow and green letters in the final guess.
(-0.02, 0.1, 0.2) by default.
- 'invalid_word_reward' - Reward given when an invalid word is entered. -1 by default.
- 'valid_word_reward' - Reward given when a valid word is entered. 0 by default.
- 'backspace_reward' - Reward given when backspace is inputted. 0 by default.
- 'step_reward' - Reward applied at every step in addition to state-specific rewards. 0 by default.
- 'truncation_limit' - If specified, will truncate each episode after this many steps.

Also optionally accepts a custom_render_settings dictionary with any of the following fields:
- 'render_mode' - Either 'command_line' or 'gui'.
- 'scale' - Factor by which to scale the window. Default is 2/3
- 'animation_duration' - Factor by which animation times are multiplied. 1 is normal speed, 0 is instant.

## Environment object methods

### reset()

Begin a new episode.

Parameters:
- word (str) - If specified, it will be set as the hidden/answer word for this episode.

Returns:
- observation (NDArray[uint8]) - Observation of the current state. See the step() method for details.
- info (dict) - Contains other information unknown to the agent. See the step() method for details.

## step(action)

Take an action.

Parameters:
- action (int or iterable containing ints) - Action encoding depends on the action mode:
    - Action mode 1 - -2 for enter, -1 for backspace or the alphabet index for a letter.
    - Action mode 2 - -1 for backspace or the alphabet index for a letter.
    - Action mode 3 - Alphabet index. The default alphabet is qwertyuiopasdfghjklzxcvbnm, so 0 = q, 1 = w and so on.
    - Action mode 4 - Iterable containing alphabet indices.
    - Action mode 5 - Index of a word in the vocab list. By default, 0 is aahed, 1 is aalii and so on.

Returns:
- observation (NDArray[uint8]) - Observation of the current state. Array has shape (2, max_guesses, word_length).
    - First dimension: 0 = letters (values correspond to alphabet indices), 1 = colours (0 for grey, 1 for
    yellow, 2 for green). Blank letter cells have a value one higher than the highest letter index.
    Empty colour cells have a value of 3.
- reward (float) - The reward earned by transitioning to the new state.
- terminal (bool) - True if the new state is a terminal state, i.e. the game is over.
- truncated (bool) - True if the episode has reached the maximum number of steps.
- info (dict) - Contains other information unknown to the agent. Has the following fields:
    - 'hidden_word' (int) - The hidden/answer word for this episode, encoded with each group of 5 bits
    representing a letter.
    - 'step' (int) - The number of steps completed so far. Always 0.
    - 'invalid_word' (bool) - True if the last action resulted in attempting to enter a word which is not on the vocab list.
    - 'incomplete_word' (bool) - True if the last action resulted in attempting to enter a row with one or more empty spaces remaining.

## render()

Displays the current state in a human-comprehensible format.

Parameters: None

Returns: None

## play()

Plays one episode, controlled by the user.

Parameters
- hidden_word (str) - If specified, it will be set as the hidden/answer word for this episode. Note that since play() automatically resets the environment, this is the only way to set the hidden word.
- keybindings (dict) - If GUI is enabled, this allows for keybindings to be customised.