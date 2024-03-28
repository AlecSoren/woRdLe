import numpy as np
import random
from collections.abc import Iterable
from numbers import Number



class Wordle_Environment:


    def reset(self, word = None):
        """
Begin a new episode.

Parameters:
- word (str) - If specified, it will be set as the hidden/answer word for this episode.

Returns:
- observation (NDArray[uint8]) - Observation of the current state. Array has shape (2, max_guesses, word_length).
    - First dimension: 0 = letters (values correspond to alphabet indices), 1 = colours (0 for grey, 1 for
    yellow, 2 for green). Empty cells have a value of 255.
- info (dict) - Contains other information unknown to the agent. Has the following fields:
    - 'hidden_word' (int) - The hidden/answer word for this episode, encoded with each group of 5 bits
    representing a letter.
    - 'step' (int) - The number of steps completed so far. Always 0.
        """

        hidden_word = []
        if word == None:
            hidden_word_code = random.choice(self.hidden_words)
            for i in range(self.word_length - 1, -1, -1):
                letter_code = (hidden_word_code & self.bitmasks[i]) >> (i * 5)
                hidden_word.append(letter_code)
        else:
            hidden_word_code = 0
            for character in word:
                letter = self.alphabet.index(character)
                hidden_word_code = hidden_word_code << 5 | letter
                hidden_word.append(letter)
        self.hidden_word_code = hidden_word_code
        self.hidden_word = hidden_word
        self.hidden_word_as_set = frozenset(hidden_word)
        hidden_word_counts = {letter: 0 for letter in self.hidden_word_as_set}
        for letter in hidden_word:
            hidden_word_counts[letter] += 1
        self.hidden_word_counts = hidden_word_counts

        self.state = np.full((2, self.max_guesses, self.word_length), 255, dtype='uint8')

        self.position = 0
        self.guess_num = 0
        self.current_row_code = 0
        self.current_row = [255] * self.word_length
        self.step_count = 0

        self.info = {
            'hidden_word': self.hidden_word_code,
            'step': 0
        }
        return (self.state, self.info)
    

    def step(self, action):
        self.step_count += 1
        self.info['step'] = self.step_count

        reward = self.step_reward
        terminal = False
        truncated = self.step_count == self.truncation_limit

        # 1 = Full control over all keys.
        # 2 = Controls letter keys and backspace, but automatically presses enter after filling row.
        # 3 = Controls letter keys only. Automatically enters after five letters and clears row if invalid.
        # 4 = Inputs one row at a time.
        # 5 = Chooses word from vocab list.
        match self.action_mode:

            case 1:

                #Backspace
                if action == 255:
                    reward += self.backspace_reward
                    if self.position != 0:
                        self.position -= 1
                        self.state[0, self.guess_num, self.position] = 255
                elif action == 254:
                    if self.position == self.word_length:
                        pass
                    else:
                        reward += self.invalid_word_reward

            case 3:
                self.state[0, self.guess_num, self.position] = action
                self.current_row[self.position] = action
                self.current_row_code = self.current_row_code << 5 | action
                self.position += 1

                #Reached the end of a row
                if self.position == self.word_length:

                    if self.current_row_code == self.hidden_word_code:
                        terminal = True
                        reward = self.correct_guess_rewards[self.guess_num]
                        self.state[1, self.guess_num] = 2

                    elif self.current_row_code in self.vocab:
                        current_row_counts = {letter: 0 for letter in self.current_row}
                        row_reward = 0
                        for i, letter in enumerate(self.current_row):
                            if letter == self.hidden_word[i]:
                                current_row_counts[letter] += 1
                        for i, letter in enumerate(self.current_row):
                            if letter == self.hidden_word[i]:
                                self.state[1, self.guess_num, i] = 2
                                row_reward += self.green_reward
                            elif letter in self.hidden_word_as_set:
                                current_row_counts[letter] += 1
                                if current_row_counts[letter] <= self.hidden_word_counts[letter]:
                                    self.state[1, self.guess_num, i] = 1
                                    row_reward += self.yellow_reward
                            else:
                                self.state[1, self.guess_num, i] = 0
                                row_reward += self.grey_reward

                        if self.guess_num == self.max_guesses - 1:
                            terminal = True
                            reward += row_reward
                        else:
                            self.guess_num += 1

                    else:
                        self.state[0, self.guess_num] = 255
                        reward += self.invalid_word_reward

                    self.position = 0
                    self.current_row_code = 0
                    self.current_row = [255] * self.word_length

                return self.state, reward, terminal, truncated, self.info


    def play(self, hidden_word = None):
        self.reset(hidden_word)
        terminal = False
        while True:
            display_string = '\n'
            for row in range(self.max_guesses):
                for position in range(self.word_length):
                    color = {0:'0', 1:'33', 2:'32',255:'0'}[self.state[1, row, position]]
                    letter = (self.alphabet + '.'*255)[self.state[0, row, position]]
                    display_string += f'\033[{color}m{letter}'
                display_string += '\n'
            display_string += '\033[0m\n'
            if terminal:
                print(display_string)
                break
            word = input(display_string)
            word = tuple(self.alphabet.index(character) for character in word)
            rewards = []
            for character in word:
                ___, reward, terminal, truncated, __ = self.step(character)
                rewards.append(str(reward))
                if terminal or truncated:
                    break



def make_word_set(filename, alphabet, word_length):
    word_list = set()
    with open(filename, 'r') as file:
        for i, line in enumerate(file):
            word = line.strip()
            if len(word) != word_length:
                raise ValueError(f'word #{i} - {word} does not match word length {word_length}')
            word_bits = 0
            for character in word:
                if character not in alphabet:
                    raise ValueError(f'word #{i} - {word} contains characters not in alphabet {alphabet}')
                word_bits <<= 5
                word_bits |= alphabet.index(character)
            word_list.add(word_bits)
    return frozenset(word_list)



def make(custom_settings = {}):
    """
Returns a Wordle environment instance. Accepts a dictionary with any of the following fields:

- 'word_length' - 5 by default. If not 2-5, you must specify a 'vocab_file' and 'hidden_words_file'.
- 'alphabet' - Valid characters the player can input. 'abcdefghijklmnopqrstuvwxyz' by default.
- 'vocab_file' - Path to a text file containing the valid word list.
- 'hidden_words_file' - Path to a text file containing all possible hidden/answer words.

- 'max_hidden_word_options' - If specified, plays with a pseudorandom subset of the hidden word list.
Gradually increase the subset size to make it easier for the agent at the start.
- 'hidden_word_subset_seed' - Determines the order in which words are chosen for the subset.
Make sure to use the same seed each time, or it will choose a new random subset of words.

- 'action_mode' - How much control over input is given to the agent. 3 by default.
    - 1 = Full control over all keys.
    - 2 = Controls letter keys and backspace, but automatically presses enter after filling row.
    - 3 = Controls letter keys only. Automatically enters after five letters and clears row if invalid.
    - 4 = Inputs one row at a time.
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
    """

    settings = {
        'word_length' : 5,
        'alphabet' : 'abcdefghijklmnopqrstuvwxyz',
        'vocab_file' : None,
        'hidden_words_file' : None,

        #If specified, chooses a random subset of the hidden word list to use for this environment instance
        #Gradually increase the subset size to make it easier for the agent at the start
        #Make sure to use the same seed each time, or it will choose a new random subset of words
        'max_hidden_word_options' : None,
        'hidden_word_subset_seed' : None,

        #1 = full control over all keys
        #2 = controls letter keys and backspace, but automatically presses enter after filling row
        #3 = controls letter keys only. Automatically enters after five letters and clears row if invalid
        #4 = inputs one row at a time
        #5 = chooses word from vocab list
        'action_mode' : 3,

        'max_guesses' : 6,

        'correct_guess_rewards' : (7, 6, 5, 4, 3, 2), #Index = how many wrong guesses before the correct one
        'final_guess_rewards' : (-0.02, 0.1, 0.2), #(grey, yellow, green)
        'invalid_word_reward' : -1,
        'valid_word_reward' : 0,
        'backspace_reward' : 0,
        'step_reward' : 0, #Reward applied at every step in addition to state-specific rewards

        'truncation_limit' : None
    }

    default_word_files = {
        2 : ('word_lists/vocab_two_letter.txt', 'word_lists/hidden_words_two_letter.txt'),
        3 : ('word_lists/vocab_three_letter.txt', 'word_lists/hidden_words_three_letter.txt'),
        4 : ('word_lists/vocab_four_letter.txt', 'word_lists/hidden_words_four_letter.txt'),
        5 : ('word_lists/vocab_five_letter.txt', 'word_lists/hidden_words_five_letter.txt')
    }

    for key in custom_settings:
        if key not in settings:
            raise KeyError(f'{key} is not a setting')
        if custom_settings[key] != None:
            settings[key] = custom_settings[key]

    #Check all fields that must be positive nonzero integers or iterables of positive nonzero integers
    for f in (
        'word_length',
        'hidden_word_subset_seed',
        'max_hidden_word_options',
        'max_guesses',
        'truncation_limit'
    ):
        value = settings[f]
        if value != None and not (isinstance(value, int) and value > 0):
            raise ValueError(f'{f} must be a positive nonzero integer')

    env = Wordle_Environment()

    word_length = settings['word_length']
    env.word_length = word_length
    env.bitmasks = tuple(31 << (i * 5) for i in range(word_length))

    alphabet = settings['alphabet']
    if not isinstance(alphabet, str):
        raise ValueError('alphabet must be a string')
    if len(set(alphabet)) != len(alphabet):
        raise ValueError('alphabet must not contain duplicate characters')
    if ''.join(alphabet.split()) != alphabet:
        raise ValueError('alphabet must not contain whitespace')
    if len(alphabet) > 31:
        raise ValueError('alphabet must not contain more than 31 characters')
    env.alphabet = alphabet
    
    if settings['vocab_file'] == None:
        if word_length not in default_word_files:
            error_message = 'there are no default vocab files for ' + str(word_length)
            error_message += '-letter words; you must provide your own'
            raise KeyError(error_message)
        else:
            vocab_file = default_word_files[word_length][0]
    else:
        if not isinstance(settings['vocab_file'], str):
            raise ValueError('vocab_file must be a string')
        vocab_file = settings['vocab_file']
    env.vocab = make_word_set(vocab_file, alphabet, word_length)

    if settings['hidden_words_file'] == None:
        if word_length not in default_word_files:
            error_message = 'there are no default hidden word files for ' + str(word_length)
            error_message += '-letter words; you must provide your own'
            raise KeyError(error_message)
        else:
            hidden_words_file = default_word_files[word_length][1]
    else:
        if not isinstance(settings['hidden_words_file'], str):
            raise ValueError('hidden_words_file must be a string')
        hidden_words_file = settings['hidden_words_file']
    hidden_words_set = make_word_set(hidden_words_file, alphabet, word_length)
    
    max_hidden_word_options = settings['max_hidden_word_options']
    if max_hidden_word_options == None:
        env.hidden_words = tuple(hidden_words_set)
    else:
        if max_hidden_word_options > len(hidden_words_set):
            raise ValueError('max_hidden_word_options is larger than the hidden word list')
        seed = settings['hidden_word_subset_seed']
        if not isinstance(seed, (type(None), int, float, str, bytes, bytearray)):
            raise ValueError('hidden_word_subset_seed must be int, float, str, bytes, or bytearray')
        hidden_words_list = list(hidden_words_set)
        if seed == None:
            random.shuffle(hidden_words_list)
        else:
            random.Random(seed).shuffle(hidden_words_list)
        env.hidden_words = tuple(hidden_words_list[:max_hidden_word_options])

    if settings['action_mode'] not in (1, 2, 3, 4, 5):
        raise ValueError('action_mode must be 1, 2, 3, 4 or 5')
    env.action_mode = settings['action_mode']

    env.max_guesses = settings['max_guesses']

    for key in ('correct_guess_rewards', 'final_guess_rewards'):
        value = settings[key]
        if not (isinstance(value, Iterable) and all(isinstance(e, Number) for e in value)):
            raise ValueError(f'{key} must be an iterable containing numbers')
        
    for key in ('invalid_word_reward', 'valid_word_reward', 'backspace_reward', 'step_reward'):
        if not isinstance(settings[key], Number):
            raise ValueError(f'{key} must be a number')
    
    env.correct_guess_rewards = settings['correct_guess_rewards']
    env.grey_reward, env.yellow_reward, env.green_reward = settings['final_guess_rewards']
    env.invalid_word_reward = settings['invalid_word_reward']
    env.valid_word_reward = settings['valid_word_reward']
    env.backspace_reward = settings['backspace_reward']
    env.step_reward = settings['step_reward']
        
    if len(env.correct_guess_rewards) != env.max_guesses:
        raise ValueError('length of correct_guess_rewards does not match max_guesses')
    
    env.truncation_limit = settings['truncation_limit']

    if env.action_mode != 3:
        raise NotImplementedError('action modes other than 3 have not been implemented yet')

    return env



if __name__ == '__main__':
    env = make()
    env.play()
