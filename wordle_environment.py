import numpy as np
import random

from collections.abc import Iterable
from numbers import Number

import pygame

import time
from math import sin



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



class Wordle_Environment:

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
        self.vocab = frozenset(make_word_list(vocab_file, alphabet, word_length))

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

        if self.action_mode != 3:
            raise NotImplementedError('action modes other than 3 have not been implemented yet')


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
            'step': 0,
            'invalid_word': False
        }
        return (self.state, self.info)
    

    def step(self, action):
        self.step_count += 1
        self.info['step'] = self.step_count
        self.info['invalid_word'] = False

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
                        self.info['invalid_word'] = True

                    self.position = 0
                    self.current_row_code = 0
                    self.current_row = [255] * self.word_length

                return self.state, reward, terminal, truncated, self.info


    def render(self):
        display_string = '\n'
        for row in range(self.max_guesses):
            for position in range(self.word_length):
                color = {0:'0', 1:'33', 2:'32',255:'0'}[self.state[1, row, position]]
                letter = (self.alphabet + '.'*255)[self.state[0, row, position]]
                display_string += f'\033[{color}m{letter}'
            display_string += '\n'
        display_string += '\033[0m'
        print(display_string)


    def play(self, hidden_word = None):
        self.reset(hidden_word)
        while True:
            self.render()
            word = input()
            word = tuple(self.alphabet.index(character) for character in word)
            rewards = []
            for character in word:
                ___, reward, terminal, truncated, __ = self.step(character)
                rewards.append(str(reward))
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
    letter_colours = [-1] * len(alphabet)
    for x in range(max_guesses):
        for y in range(word_length):
            letter_code, colour_code = state[:, x, y]
            if colour_code == 255:
                return letter_colours
            letter_colours[letter_code] = max(colour_code, letter_colours[letter_code])
    return letter_colours



def draw_square(screen, font, scale, center_coords, letter = None, colour_code = 3):
    rect = pygame.Rect(
        center_coords[0] - 39 * scale,
        center_coords[1] - 39 * scale,
        78 * scale,
        78 * scale
        )
    if letter == None:
        pygame.draw.rect(screen, (211, 214, 218), rect, 3)
    else:
        try:
            colour_value = ((120, 124, 126), (201, 180, 88), (106, 170, 100))[colour_code]
        except IndexError:
            colour_value = (135, 138, 140)
        pygame.draw.rect(screen, colour_value, rect)
        img = font.render(letter.upper(), True, (255, 255, 255))
        rect = img.get_rect()
        rect.center = (center_coords)
        screen.blit(img, rect)



class Wordle_GUI_Wrapper:

    def __init__(self, env = None, custom_render_settings = {}):

        self.env = env
        env.reset()
    
        render_settings = {
            'render_mode':'gui', #'command_line' or 'gui',
            'display_scale': 2/3,
            'animation_duration': 1.0 #1 is normal speed, 0 is instant
        }

        override_settings(render_settings, custom_render_settings)

        for k in ('display_scale', 'animation_duration'):
            value = render_settings[k]
            if (not isinstance(value, Number)) or value <= 0:
                raise ValueError(f'{k} must be a positive number')

        self.scale = render_settings['display_scale']
        self.animation_duration = render_settings['animation_duration']

        self.render()

    
    def reset(self, word = None):
        return self.env.reset(word)


    def step(self, action):
        result = self.env.step(action)
        if not self.currently_rendered:
            return result
        
        state, reward, terminal, truncated, info = result

        scale = self.scale
        env = self.env
        screen = self.screen

        #Add letters
        new_letters = []
        

        #Invalid word animation
        if info['invalid_word']:
            for row in range(env.max_guesses):
                if state[1, row, 0] == 255:
                    current_row = row
                    break

            square_colours = state[1, current_row]
            letters = [None] * env.word_length
            for i, l in enumerate(state[0, current_row]):
                if l != 255:
                    letters[i] = l

            anim_time = 0.62 * self.animation_duration
            max_offset = 7
            shakes = 10

            background_rect = pygame.Rect(
                scale * (self.x_origin - 78 / 2 - max_offset),
                scale * (105 + 86 * current_row),
                scale * (env.word_length * 85.5 + 78 + max_offset * 2),
                scale * 94
                )
            row_coords = self.board_coords[current_row]

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
        
        self.render()
        return result


    def render(self):
        scale = self.scale
        env = self.env
        alphabet = env.alphabet

        self.currently_rendered = True
        self.old_state = env.state

        pygame.init()
        screen_width = max(880, 85.5 * env.word_length + 15) #Normally 880
        screen_height = 86 * env.max_guesses + 484 #Normally 1000
        self.screen_width, self.screen_height = screen_width, screen_height
        screen = pygame.display.set_mode((scale * screen_width, scale * screen_height))
        self.screen = screen
        pygame.display.set_caption('Wordle Environment')
        screen.fill((255, 255, 255))

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

        keyboard_font = pygame.font.SysFont(None, round(scale * 42))
        letter_colours = get_letter_colours (
            env.state, env.max_guesses, env.word_length, alphabet
            )
        self.key_coords = [key_coords(i, scale, screen_width, screen_height) for i in range(len(alphabet))]
        for i, colour in enumerate(letter_colours):
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
            img = keyboard_font.render(alphabet[i].upper(), True, letter_colour)
            rect = img.get_rect()
            rect.center = (center_coords)
            screen.blit(img, rect)

        pygame.display.update()  


    def stop_render(self):
        self.currently_rendered = False
        pygame.quit()


    def play(self, hidden_word = None, keybindings = None):

        if keybindings == None:
            keybindings = {i+97:'qwertyuiopasdfghjklzxcvbnm'.index(l)
                            for i, l in enumerate('abcdefghijklmnopqrstuvwxyz')}
            
        self.reset(hidden_word)
        self.render()
        terminal, truncated = False, False
        while True:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    try:
                        action = keybindings[event.key]
                    except KeyError:
                        pass
                    else:
                        state, reward, terminal, truncated, info = self.step(action)
                        self.render()
            if terminal or truncated:
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
    env = make(custom_render_settings={'render_mode':'gui'})
    env.play()
