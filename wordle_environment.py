import numpy as np
import random

from collections.abc import Iterable
from numbers import Number

import pygame

import time
from math import sin, cos



def override_settings(default_settings, custom_settings):
    for key in custom_settings:
        if key not in default_settings:
            raise KeyError(f'{key} is not a setting')
        if custom_settings[key] != None:
            default_settings[key] = custom_settings[key]



def make_word_list(filename, alphabet, word_length):
    word_list = []
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
            word_list.append(word_bits)
    return word_list



def bits_to_word(bits, alphabet):
    word = []
    while bits != 0:
        character_bits = bits & 31
        word.insert(0, alphabet[character_bits])
        bits >>= 5
    return ''.join(word)



class Wordle_Environment:

    # 1 = Full control over all keys. Backspace: -1, enter: -2
    # 2 = Controls letter keys and backspace, but automatically presses enter after filling row.
    # 3 = Controls letter keys only. Automatically enters after five letters and clears row if invalid.
    # 4 = Inputs one row at a time.
    # 5 = Chooses word from vocab list.

    def __step_mode_1(self, action):
        self.step_count += 1
        self.info['step'] = self.step_count
        self.info['invalid_word'] = False
        self.info['incomplete_word'] = False

        reward = self.step_reward
        terminal = False
        truncated = self.step_count == self.truncation_limit

        #Backspace
        if action == -1:
            reward += self.backspace_reward
            if self.position != 0:
                self.position -= 1
                self.state[0, self.guess_num, self.position] = self.blank_letter_number
                self.current_row[self.position] = self.blank_letter_number
                self.current_row_code >>= 5

        #Enter
        elif action == -2:

            #At the end of a row - make a guess
            if self.position == self.word_length:

                if self.current_row_code == self.hidden_word_code:
                    terminal = True
                    reward += self.correct_guess_rewards[self.guess_num] + self.valid_word_reward
                    self.state[1, self.guess_num] = 2

                    self.position = 0
                    self.current_row_code = 0
                    self.current_row = [self.blank_letter_number] * self.word_length

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

                        else:
                            self.state[1, self.guess_num, i] = 0
                            row_reward += self.grey_reward

                    if self.guess_num == self.max_guesses - 1:
                        terminal = True
                        reward += row_reward
                    else:
                        self.guess_num += 1
                        self.position = 0
                        self.current_row_code = 0
                        self.current_row = [self.blank_letter_number] * self.word_length
                    reward += self.valid_word_reward

                else:
                    reward += self.invalid_word_reward
                    self.info['invalid_word'] = True

            #Word is incomplete
            else:
                reward += self.invalid_word_reward
                self.info['incomplete_word'] = True

        #Letter
        elif self.position != self.word_length:
            self.state[0, self.guess_num, self.position] = action
            self.current_row[self.position] = action
            self.current_row_code = self.current_row_code << 5 | action
            self.position += 1

        return self.state, reward, terminal, truncated, self.info


    def __step_mode_2(self, action):
        self.step_count += 1
        self.info['step'] = self.step_count
        self.info['invalid_word'] = False

        reward = self.step_reward
        terminal = False
        truncated = self.step_count == self.truncation_limit

        #Backspace
        if action == -1:
            reward += self.backspace_reward
            if self.position != 0:
                self.position -= 1
                self.state[0, self.guess_num, self.position] = self.blank_letter_number
                self.current_row[self.position] = self.blank_letter_number
                self.current_row_code >>= 5

        #Letter
        elif self.position != self.word_length:
            self.state[0, self.guess_num, self.position] = action
            self.current_row[self.position] = action
            self.current_row_code = self.current_row_code << 5 | action
            self.position += 1

        #Reached the end of a row
        if self.position == self.word_length:

            if self.current_row_code == self.hidden_word_code:
                terminal = True
                reward += self.correct_guess_rewards[self.guess_num] + self.valid_word_reward
                self.state[1, self.guess_num] = 2

                self.position = 0
                self.current_row_code = 0
                self.current_row = [self.blank_letter_number] * self.word_length

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

                    else:
                        self.state[1, self.guess_num, i] = 0
                        row_reward += self.grey_reward

                if self.guess_num == self.max_guesses - 1:
                    terminal = True
                    reward += row_reward
                else:
                    self.guess_num += 1
                    self.position = 0
                    self.current_row_code = 0
                    self.current_row = [self.blank_letter_number] * self.word_length

                reward += self.valid_word_reward

            else:
                reward += self.invalid_word_reward
                self.info['invalid_word'] = True

        return self.state, reward, terminal, truncated, self.info


    def __step_mode_3(self, action):
        self.step_count += 1
        self.info['step'] = self.step_count
        self.info['invalid_word'] = False

        reward = self.step_reward
        terminal = False
        truncated = self.step_count == self.truncation_limit

        self.state[0, self.guess_num, self.position] = action
        self.current_row[self.position] = action
        self.current_row_code = self.current_row_code << 5 | action
        self.position += 1

        #Reached the end of a row
        if self.position == self.word_length:

            if self.current_row_code == self.hidden_word_code:
                terminal = True
                reward += self.correct_guess_rewards[self.guess_num] + self.valid_word_reward
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

                    else:
                        self.state[1, self.guess_num, i] = 0
                        row_reward += self.grey_reward

                if self.guess_num == self.max_guesses - 1:
                    terminal = True
                    reward += row_reward
                else:
                    self.guess_num += 1

                reward += self.valid_word_reward

            else:
                self.state[0, self.guess_num] = self.blank_letter_number
                reward += self.invalid_word_reward
                self.info['invalid_word'] = True

            self.position = 0
            self.current_row_code = 0
            self.current_row = [self.blank_letter_number] * self.word_length

        return self.state, reward, terminal, truncated, self.info


    def __step_mode_4(self, action):
        self.step_count += 1
        self.info['step'] = self.step_count
        self.info['invalid_word'] = False

        reward = self.step_reward
        terminal = False
        truncated = self.step_count == self.truncation_limit

        #Calculate row code
        row_code = 0
        for l in action:
            row_code = row_code << 5 | l
        
        #Correct guess
        if row_code == self.hidden_word_code:
            terminal = True
            reward += self.correct_guess_rewards[self.guess_num] + self.valid_word_reward
            self.state[0, self.guess_num] = action
            self.state[1, self.guess_num] = 2

        elif row_code in self.vocab:
            self.state[0, self.guess_num] = action
            current_row_counts = {letter: 0 for letter in action}
            row_reward = 0
            for i, letter in enumerate(action):
                if letter == self.hidden_word[i]:
                    current_row_counts[letter] += 1
            for i, letter in enumerate(action):
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

                else:
                    self.state[1, self.guess_num, i] = 0
                    row_reward += self.grey_reward

            if self.guess_num == self.max_guesses - 1:
                terminal = True
                reward += row_reward
            else:
                self.guess_num += 1

            reward += self.valid_word_reward

        else:
            self.info['invalid_word'] = True
            reward += self.invalid_word_reward

        return self.state, reward, terminal, truncated, self.info


    def __step_mode_5(self, action):
        self.step_count += 1
        self.info['step'] = self.step_count

        reward = self.step_reward + self.valid_word_reward
        terminal = False
        truncated = self.step_count == self.truncation_limit

        row_code = self.vocab_list[action]
        word = []
        for i in range(self.word_length - 1, -1, -1):
            letter_code = (row_code & self.bitmasks[i]) >> (i * 5)
            word.append(letter_code)
        
        #Correct guess
        if row_code == self.hidden_word_code:
            terminal = True
            reward += self.correct_guess_rewards[self.guess_num]
            self.state[0, self.guess_num] = word
            self.state[1, self.guess_num] = 2

        else:
            self.state[0, self.guess_num] = word
            current_row_counts = {letter: 0 for letter in word}
            row_reward = 0
            for i, letter in enumerate(word):
                if letter == self.hidden_word[i]:
                    current_row_counts[letter] += 1
            for i, letter in enumerate(word):
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

                else:
                    self.state[1, self.guess_num, i] = 0
                    row_reward += self.grey_reward

            if self.guess_num == self.max_guesses - 1:
                terminal = True
                reward += row_reward
            else:
                self.guess_num += 1

        return self.state, reward, terminal, truncated, self.info


    def __init__(self, custom_settings = {}):

        settings = {
            'word_length' : 5,
            'alphabet' : 'qwertyuiopasdfghjklzxcvbnm',
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

        override_settings(settings, custom_settings)

        default_word_files = {
            2 : ('word_lists/vocab_two_letter.txt', 'word_lists/hidden_words_two_letter.txt'),
            3 : ('word_lists/vocab_three_letter.txt', 'word_lists/hidden_words_three_letter.txt'),
            4 : ('word_lists/vocab_four_letter.txt', 'word_lists/hidden_words_four_letter.txt'),
            5 : ('word_lists/vocab_five_letter.txt', 'word_lists/hidden_words_five_letter.txt')
        }

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

        word_length = settings['word_length']
        self.word_length = word_length
        self.bitmasks = tuple(31 << (i * 5) for i in range(word_length))

        alphabet = settings['alphabet']
        if not isinstance(alphabet, str):
            raise ValueError('alphabet must be a string')
        if len(set(alphabet)) != len(alphabet):
            raise ValueError('alphabet must not contain duplicate characters')
        if ''.join(alphabet.split()) != alphabet:
            raise ValueError('alphabet must not contain whitespace')
        if len(alphabet) > 26:
            raise ValueError('alphabet must not contain more than 26 characters')
        self.alphabet = alphabet
        self.blank_letter_number = len(alphabet)
        
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
        self.vocab_list = make_word_list(vocab_file, alphabet, word_length)
        self.vocab = frozenset(self.vocab_list)

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
        hidden_words_list = make_word_list(hidden_words_file, alphabet, word_length)
        
        max_hidden_word_options = settings['max_hidden_word_options']
        if max_hidden_word_options == None:
            self.hidden_words = tuple(hidden_words_list)
        else:
            if max_hidden_word_options > len(hidden_words_list):
                raise ValueError('max_hidden_word_options is larger than the hidden word list')
            seed = settings['hidden_word_subset_seed']
            if not isinstance(seed, (type(None), int, float, str, bytes, bytearray)):
                raise ValueError('hidden_word_subset_seed must be int, float, str, bytes, or bytearray')
            if seed == None:
                random.shuffle(hidden_words_list)
            else:
                random.Random(seed).shuffle(hidden_words_list)
            self.hidden_words = tuple(hidden_words_list[:max_hidden_word_options])

        if settings['action_mode'] not in (1, 2, 3, 4, 5):
            raise ValueError('action_mode must be 1, 2, 3, 4 or 5')
        self.action_mode = settings['action_mode']
        self.step = (
            self.__step_mode_1,
            self.__step_mode_2,
            self.__step_mode_3,
            self.__step_mode_4,
            self.__step_mode_5
        )[self.action_mode - 1]

        self.max_guesses = settings['max_guesses']

        for key in ('correct_guess_rewards', 'final_guess_rewards'):
            value = settings[key]
            if not (isinstance(value, Iterable) and all(isinstance(e, Number) for e in value)):
                raise ValueError(f'{key} must be an iterable containing numbers')
            
        for key in ('invalid_word_reward', 'valid_word_reward', 'backspace_reward', 'step_reward'):
            if not isinstance(settings[key], Number):
                raise ValueError(f'{key} must be a number')
        
        self.correct_guess_rewards = settings['correct_guess_rewards']
        self.grey_reward, self.yellow_reward, self.green_reward = settings['final_guess_rewards']
        self.invalid_word_reward = settings['invalid_word_reward']
        self.valid_word_reward = settings['valid_word_reward']
        self.backspace_reward = settings['backspace_reward']
        self.step_reward = settings['step_reward']
            
        if len(self.correct_guess_rewards) != self.max_guesses:
            raise ValueError('length of correct_guess_rewards does not match max_guesses')
        
        self.truncation_limit = settings['truncation_limit']

        #if self.action_mode != 3:
        #    raise NotImplementedError('action modes other than 3 have not been implemented yet')


    def reset(self, word = None):
        """
Begin a new episode.

Parameters:
- word (str) - If specified, it will be set as the hidden/answer word for this episode.

Returns:
- observation (NDArray[uint8]) - Observation of the current state. Array has shape (2, max_guesses, word_length).
    - First dimension: 0 = letters (values correspond to alphabet indices), 1 = colours (0 for grey, 1 for
    yellow, 2 for green). Blank letter cells are one higher than the highest letter index.
    Empty colour cells have a value of 3.
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

        subarray_shape = (self.max_guesses, self.word_length)
        self.state = np.stack((
            np.full(subarray_shape, self.blank_letter_number, dtype='uint8'),
            np.full(subarray_shape, 3, dtype='uint8')
        ))

        self.position = 0
        self.guess_num = 0
        self.current_row_code = 0
        self.current_row = [self.blank_letter_number] * self.word_length
        self.step_count = 0

        self.info = {
            'hidden_word': self.hidden_word_code,
            'step': 0,
            'invalid_word': False,
            'incomplete_word': False
        }
        return (self.state, self.info)
    

    def render(self):
        display_string = '\n'
        for row in range(self.max_guesses):
            for position in range(self.word_length):
                color = ['0', '33', '32', '0'][self.state[1, row, position]]
                letter = (self.alphabet + '.')[self.state[0, row, position]]
                display_string += f'\033[{color}m{letter}'
            display_string += '\n'
        display_string += '\033[0m'
        print(display_string)


    def play(self, hidden_word = None):
        self.reset(hidden_word)
        while True:
            self.render()

            user_input = input()

            if self.action_mode <= 3:
                actions = []
                for c in user_input:
                    if c in ('1', '2'):
                        actions.append(- int(c))
                    else:
                        actions.append(self.alphabet.index(c))

            elif self.action_mode == 4:
                actions = (tuple(self.alphabet.index(character) for character in user_input),)

            elif self.action_mode == 5:
                actions = (int(user_input),)

            rewards = []
            for a in actions:
                ___, reward, terminal, truncated, __ = self.step(a)
                rewards.append(str(reward))
                if terminal or truncated:
                    break
            print(' '.join(rewards))
            if terminal or truncated:
                self.render()
                break



def key_coords(index, scale, screen_width, screen_height):
    if index <= 9:
        x, y = 109.5 + 74 * index, 681.5
    elif index <= 18:
        x, y = 142.5 + 74 * (index - 10), 780.5
    else:
        x, y = 216.5 + 74 * (index - 19), 879.5
    return (scale * ((screen_width - 880) / 2 + x), scale * (screen_height - 1000 + y))
    


def get_letter_colours(state, max_guesses, word_length, alphabet):
    letter_colours = [3] * len(alphabet)
    for x in range(max_guesses):
        for y in range(word_length):
            letter_code, colour_code = state[:, x, y]
            if colour_code == 3:
                return letter_colours
            letter_colours[letter_code] = max(colour_code, letter_colours[letter_code])
    return letter_colours



def draw_square(
        screen, font, scale, center_coords,
        letter = None, colour_code = 3, x_scale = 1, y_scale = 1, erase = False
        ):
    x_scale_factor = scale * x_scale
    y_scale_factor = scale * y_scale
    rect = pygame.Rect(
        center_coords[0] - 39 * x_scale_factor,
        center_coords[1] - 39 * y_scale_factor,
        78 * x_scale_factor,
        78 * y_scale_factor
        )
    if erase:
        pygame.draw.rect(screen, (255, 255, 255), rect)
    elif letter == None:
        pygame.draw.rect(screen, (255, 255, 255), rect)
        pygame.draw.rect(screen, (211, 214, 218), rect, round(4.5 * scale))
    elif colour_code == 3:
        pygame.draw.rect(screen, (135, 138, 140), rect, round(4.5 * scale))
    else:
        colour_value = ((120, 124, 126), (201, 180, 88), (106, 170, 100))[colour_code]
        pygame.draw.rect(screen, colour_value, rect)

    if letter != None:
        if colour_code == 3:
            letter_colour = (0, 0, 0)
        else:
            letter_colour = (255, 255, 255)
        img = font.render(letter.upper(), True, letter_colour)
        img = pygame.transform.scale_by(img, (x_scale, y_scale))
        rect = img.get_rect()
        rect.center = (center_coords)
        screen.blit(img, rect)



def draw_message(screen, scale, font, message = None):

    screen_width = screen.get_size()[0]

    if message == None:
        #Erase previous message
        rect = pygame.Rect(0, 0, screen_width, 90 * scale)
        pygame.draw.rect(screen, (255, 255, 255), rect)

    else:
        img = font.render(message, True, (255, 255, 255))

        coords = (screen_width / 2, 57 * scale)

        #Draw box
        rect = pygame.Rect(0, 0, img.get_size()[0] + 40 * scale, 63 * scale)
        rect.center = coords
        r = round(4 * scale)
        pygame.draw.rect(screen, (0, 0, 0), rect, 0, r, r, r, r)

        #Display message
        rect = img.get_rect()
        rect.center = (coords)
        screen.blit(img, rect)



class Wordle_GUI_Wrapper:

    def __initalise_pygame(self):
        screen_width = max(880, 85.5 * self.env.word_length + 15) #Normally 880
        screen_height = 86 * self.env.max_guesses + 484 #Normally 1000
        self.screen_width, self.screen_height = screen_width, screen_height

        pygame.init()
        screen = pygame.display.set_mode((self.scale * screen_width, self.scale * screen_height))
        self.screen = screen
        pygame.display.set_caption('Wordle Environment')
        screen.fill((255, 255, 255))
        pygame.display.update()


    def __init__(self, env = None, custom_render_settings = {}):

        self.env = env
        env.reset()
    
        render_settings = {
            'render_mode':'gui', #'command_line' or 'gui',
            'scale': 2/3,
            'animation_duration': 1.0 #1 is normal speed, 0 is instant
        }

        override_settings(render_settings, custom_render_settings)

        for k in ('scale', 'animation_duration'):
            value = render_settings[k]
            if (not isinstance(value, Number)) or value <= 0:
                raise ValueError(f'{k} must be a positive number')

        self.scale = render_settings['scale']
        self.animation_duration = render_settings['animation_duration']

        self.success_messages = ('Genius', 'Magnificent', 'Impressive', 'Splendid')[:env.max_guesses]
        self.success_messages += ('Great',) * (env.max_guesses - 5)
        if env.max_guesses >= 6:
            self.success_messages += ('Phew',)

        self.render()

    
    def reset(self, word = None):
        result = self.env.reset(word)
        if self.currently_rendered:
            self.render()
        return result


    def step(self, action):
        if not self.currently_rendered:
            return self.env.step(action)
        
        scale = self.scale
        env = self.env
        screen = self.screen

        old_state = np.copy(env.state)
        guess_num, position = env.guess_num, env.position

        state, reward, terminal, truncated, info = env.step(action)

        #Action is a single letter
        if self.env.action_mode <= 3 and action >= 0 and action <= self.env.blank_letter_number:
            added_letters = (env.alphabet[action],)
        #Action is a sequence of letters
        elif self.env.action_mode == 4:
            added_letters = (env.alphabet[l] for l in action)
        #Action is the index of a word in the vocab list
        elif self.env.action_mode == 5:
            added_letters = bits_to_word(env.vocab_list[action], env.alphabet)
        #Action is backspace or enter
        else:
            added_letters = None
        
        #Add letters
        if added_letters != None and position != env.word_length:
            row_coords = self.board_coords[guess_num]
            for letter in added_letters:
                coords = row_coords[position]
                scale_up = 1
                anim_time = 0.11333333333 * self.animation_duration
                finish_time = time.time() + anim_time
                while True:
                    draw_square(
                        screen, None, scale, coords, x_scale = scale_up, y_scale = scale_up, erase = True
                        )
                    scale_up = max(sin((finish_time - time.time()) / anim_time * 3.14), 0) * 0.12 + 1
                    draw_square(
                        screen, self.board_font, scale, coords, letter,
                        x_scale = scale_up, y_scale = scale_up
                        )
                    pygame.display.update()
                    if time.time() >= finish_time:
                        break

                old_state[0, guess_num, position] = env.alphabet.index(letter)
                position += 1

        clear_keypress_buffer = False

        #Invalid word animation
        if info['invalid_word'] or info['incomplete_word']:
            clear_keypress_buffer = True

            if info['invalid_word']:
                message = 'Not in word list'
            else:
                message = 'Not enough letters'
            draw_message(screen, scale, self.message_font, message)

            square_colours = old_state[1, guess_num]
            letters = [None] * env.word_length
            for i, l in enumerate(old_state[0, guess_num]):
                if l != self.env.blank_letter_number:
                    letters[i] = env.alphabet[l]

            anim_time = 0.62 * self.animation_duration
            max_offset = 7
            shakes = 10

            background_rect = pygame.Rect(
                scale * (self.x_origin - 78 / 2 - max_offset),
                scale * (105 + 86 * guess_num),
                scale * (env.word_length * 85.5 + 78 + max_offset * 2),
                scale * 94
                )
            row_coords = self.board_coords[guess_num]

            start_time = time.time()
            finish_time = start_time + anim_time
            while True:
                completion = min(1, (time.time() - start_time) / anim_time)
                radians = completion * 3.14
                offset = sin(radians * shakes) * sin(radians) * max_offset

                pygame.draw.rect(screen, (255, 255, 255), background_rect)
                for i, center_coords in enumerate(row_coords):
                    new_coords = (center_coords[0] + scale * offset, center_coords[1])
                    draw_square(screen, self.board_font, scale, new_coords, letters[i], square_colours[i])

                pygame.display.update()

                if time.time() >= finish_time:
                    break

            draw_message(screen, scale, self.message_font)

        correct_answer = all(state[1, guess_num] == 2)

        #Flip letters, revealing colours
        if state[1, guess_num, 0] != old_state[1, guess_num, 0]:
            clear_keypress_buffer = True

            if guess_num + 1 == env.max_guesses and not correct_answer:
                draw_message(screen, scale, self.message_font,
                             bits_to_word(env.hidden_word_code, env.alphabet).upper())
                
            changed_key_colours = []

            row_coords = self.board_coords[guess_num]
            for i in range(env.word_length):
                l = state[0, guess_num, i]
                letter = env.alphabet[l]
                colour = state[1, guess_num, i]
                coords = row_coords[i]

                anim_time = self.animation_duration / 3
                y_scale = 1
                finish_time = time.time() + anim_time
                while True:
                    draw_square(screen, None, scale, coords,
                                x_scale = 1.1, y_scale = y_scale * 1.1, erase = True)
                    completion = max((finish_time - time.time()) / anim_time, 0)
                    if completion > 0.5:
                        colour_code = 3
                    else:
                        colour_code = colour
                    y_scale = abs(cos(completion * 3.14))
                    draw_square(screen, self.board_font, scale, coords, letter, colour_code, y_scale = y_scale)
                    pygame.display.update()
                    if time.time() >= finish_time:
                        draw_square(screen, self.board_font, scale, coords, letter, colour, y_scale = 1)
                        pygame.display.update()
                        break

                if (colour + 1) % 4 > (self.letter_colours[l] + 1) % 4:
                    self.letter_colours[l] = colour
                    changed_key_colours.append(l)

            #Change keyboard colours
            for l in changed_key_colours:
                letter = env.alphabet[l]
                c = self.letter_colours[l]
                coords = self.key_coords[l]

                #Erase previous key colour
                slightly_bigger_scale = scale * 1.1
                bg_rect = (
                    coords[0] - 32.5 * slightly_bigger_scale,
                    coords[1] - 43.5 * slightly_bigger_scale,
                    65 * slightly_bigger_scale,
                    87 * slightly_bigger_scale
                )
                pygame.draw.rect(screen, (255, 255, 255), bg_rect)

                #Draw background rectangle
                rectangle_colour = ((120, 124, 126), (201, 180, 88), (106, 170, 100), (211, 214, 218))[c]
                bg_rect = (
                    coords[0] - 32.5 * scale,
                    coords[1] - 43.5 * scale,
                    65 * scale,
                    87 * scale
                )
                r = round(4 * scale)
                pygame.draw.rect(screen, rectangle_colour, bg_rect, 0, r, r, r, r)

                #Draw letter
                letter_colour = ((255, 255, 255), (255, 255, 255), (255, 255, 255), 0)[c]
                img = self.keyboard_font.render(letter.upper(), True, letter_colour)
                rect = img.get_rect()
                rect.center = (coords)
                screen.blit(img, rect)

            draw_message(screen, scale, self.message_font)

        #Move letters up and down to celebrate if the player gets the right answer
        if correct_answer:

            row_coords = self.board_coords[guess_num]
            letters = [env.alphabet[l] for l in state[0, guess_num]]

            row_above_exists = guess_num != 0
            if row_above_exists:
                row_above_coords = self.board_coords[guess_num - 1]
                row_above_letters = [env.alphabet[l] for l in state[0, guess_num - 1]]
                row_above_colours = state[1, guess_num - 1]

            anim_time = (5/6 + 0.1 * (env.word_length - 1)) * self.animation_duration
            offsets = [(0.1 * i * self.animation_duration) / anim_time for i in range(env.word_length)]
            finish_time = time.time() + anim_time
            while True:
                completion = 1 - max((finish_time - time.time()) / anim_time, 0)
                for i in range(env.word_length):

                    #Erase previous image
                    rect = pygame.Rect(
                    row_coords[i][0] - 39 * scale,
                    row_coords[i][1] - 121 * scale,
                    78 * scale,
                    165 * scale
                    )
                    pygame.draw.rect(screen, (255, 255, 255), rect)

                    if row_above_exists:
                        draw_square(
                            screen, self.board_font, scale, row_above_coords[i], row_above_letters[i],
                            row_above_colours[i]
                        )

                    cell_completion = max(completion * (1 + offsets[i]) - offsets[i], 0)
                    sin_value = sin((cell_completion * 6 - 0.5) * 3.14) + 1
                    coords = (
                        row_coords[i][0],
                        row_coords[i][1] - (sin_value * 37 * (1 - cell_completion) ** 2) * scale
                    )
                    draw_square(screen, self.board_font, scale, coords, letters[i], 2)
                draw_message(screen, scale, self.message_font, self.success_messages[guess_num])
                pygame.display.update()
                if completion == 1:
                    break
                
            draw_message(screen, scale, self.message_font)
        
        #Remove deleted letters
        for i in range(env.word_length):
            if state[0, guess_num, i] != old_state[0, guess_num, i]:
                draw_square(screen, None, scale, self.board_coords[guess_num][i])
        pygame.display.update()

        self.keypress_buffer = pygame.event.get()
        for event in self.keypress_buffer:
            if event.type == pygame.QUIT:
                self.stop_render()
        if clear_keypress_buffer:
            self.keypress_buffer = []

        return state, reward, terminal, truncated, info


    def render(self):
        scale = self.scale
        env = self.env
        alphabet = env.alphabet

        self.currently_rendered = True
        self.old_state = env.state
        self.keypress_buffer = []

        self.__initalise_pygame()
        screen_width, screen_height = self.screen_width, self.screen_height
        screen = self.screen

        self.backspace_icon = pygame.image.load('gui_images/backspace_icon.png').convert()
        self.backspace_icon = pygame.transform.smoothscale_by(self.backspace_icon, self.scale)

        enter_icon_font = pygame.font.SysFont(None, round(self.scale * 26))
        self.enter_icon = enter_icon_font.render('ENTER', True, (0, 0, 0))

        board_font = pygame.font.SysFont(None, round(scale * 68))
        self.board_font = board_font
        x_origin = (screen_width - 85.5 * (env.word_length - 1)) / 2
        self.x_origin = x_origin
        board_coords = []
        for r_pos in range(env.max_guesses):
            row_coords = []
            for l_pos in range(env.word_length):
                center_coords = (scale * (x_origin + l_pos * 85.5), scale * (152 + r_pos * 86))
                row_coords.append(center_coords)
                letter_code = env.state[0, r_pos, l_pos]
                try:
                    letter = alphabet[letter_code]
                except IndexError:
                    draw_square(screen, board_font, scale, center_coords)
                else:
                    colour_code = env.state[1, r_pos, l_pos]
                    draw_square(screen, board_font, scale, center_coords, letter, colour_code)
            board_coords.append(row_coords)
        self.board_coords = board_coords

        self.keyboard_font = pygame.font.SysFont(None, round(scale * 42))
        self.letter_colours = get_letter_colours (
            env.state, env.max_guesses, env.word_length, alphabet
            )
        self.key_coords = [key_coords(i, scale, screen_width, screen_height) for i in range(len(alphabet))]
        for i, colour in enumerate(self.letter_colours):
            center_coords = self.key_coords[i]

            rectangle_colour = ((120, 124, 126), (201, 180, 88), (106, 170, 100), (211, 214, 218))[colour]
            bg_rect = (
                    center_coords[0] - 32.5 * scale,
                    center_coords[1] - 43.5 * scale,
                    65 * scale,
                    87 * scale
                    )
            r = round(4 * scale)
            pygame.draw.rect(screen, rectangle_colour, bg_rect, 0, r, r, r, r)
            
            letter_colour = ((255, 255, 255), (255, 255, 255), (255, 255, 255), 0)[colour]
            img = self.keyboard_font.render(alphabet[i].upper(), True, letter_colour)
            rect = img.get_rect()
            rect.center = (center_coords)
            screen.blit(img, rect)

        #Backspace key
        if env.action_mode <= 2:
            box_rect = pygame.Rect(
                    (screen_width / 2 + 261) * scale,
                    (screen_height - 164) * scale,
                    98 * scale,
                    87 * scale
                    )
            r = round(4 * scale)
            pygame.draw.rect(screen, (211, 214, 218), box_rect, 0, r, r, r, r)

            img_rect = self.backspace_icon.get_rect()
            img_rect.center = box_rect.center
            screen.blit(self.backspace_icon, img_rect)

        #Enter key
        if env.action_mode == 1:
            box_rect = pygame.Rect(
                    (screen_width / 2 - 367) * scale,
                    (screen_height - 164) * scale,
                    98 * scale,
                    87 * scale
                    )
            r = round(4 * scale)
            pygame.draw.rect(screen, (211, 214, 218), box_rect, 0, r, r, r, r)

            img_rect = self.enter_icon.get_rect()
            img_rect.center = box_rect.center
            screen.blit(self.enter_icon, img_rect)

        pygame.display.update()

        self.message_font = pygame.font.SysFont('franklingothicdemi', round(scale * 22))


    def stop_render(self):
        self.currently_rendered = False
        pygame.quit()


    def play(self, hidden_word = None, keybindings = None):

        if keybindings == None:
            if self.env.action_mode <= 4:
                keybindings = {
                    i+97:'qwertyuiopasdfghjklzxcvbnm'.index(l) for i, l in enumerate('abcdefghijklmnopqrstuvwxyz')
                }
                if self.env.action_mode <= 2:
                    keybindings[8] = -1
                    if self.env.action_mode == 1:
                        keybindings[13] = -2
            if self.env.action_mode == 5:
                keybindings = {i+48:i for i in range(10)}
                keybindings[13] = -2
            
        self.reset(hidden_word)
        self.render()
        terminal, truncated = False, False

        action_buffer = []
        action_counter = 0

        while True:
            events = [e for e in pygame.event.get()] + self.keypress_buffer
            self.keypress_buffer = []
            for event in events:
                if event.type == pygame.KEYDOWN:
                    try:
                        action = keybindings[event.key]
                    except KeyError:
                        pass
                    else:

                        if self.env.action_mode <= 3:
                            state, reward, terminal, truncated, info = self.step(action)

                        elif self.env.action_mode == 4:
                            action_buffer.append(action)
                            if len(action_buffer) == self.env.word_length:
                                state, reward, terminal, truncated, info = self.step(action_buffer)
                                action_buffer = []

                        elif self.env.action_mode == 5:
                            if action == -2:
                                state, reward, terminal, truncated, info = self.step(action_counter)
                                action_counter = 0
                            else:
                                action_counter = action_counter * 10 + action

                elif event.type == pygame.QUIT:
                    self.stop_render()
                    return
                
            if terminal or truncated or not self.currently_rendered:
                break



def make(custom_settings = {}, custom_render_settings = {}):
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
    
    env = Wordle_Environment(custom_settings)

    if 'render_mode' in custom_render_settings:
        render_mode = custom_render_settings['render_mode']
        if render_mode == 'gui':
            return Wordle_GUI_Wrapper(env, custom_render_settings)
        elif render_mode == 'command_line':
            return env
        else:
            raise ValueError(f'there is no such render_mode as {render_mode}')
        
    return env



if __name__ == '__main__':
    env = make(custom_render_settings = {'render_mode':'gui'})
    env.play()
