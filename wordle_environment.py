import numpy as np
from random import choice



class Wordle_Environment:

    def __init__(self):
        self.allowed_words = set()
        with open("vocab_five_letter.txt", 'r') as file:
            for line in file:
                word = tuple('abcdefghijklmnopqrstuvwxyz'.index(character) for character in line.strip().lower())
                self.allowed_words.add(word)

        self.hidden_words = set()
        with open("hidden_words_five_letter.txt", 'r') as file:
            for line in file:
                word = tuple('abcdefghijklmnopqrstuvwxyz'.index(character) for character in line.strip().lower())
                self.hidden_words.add(word)

        self.reset()


    def reset(self, word = None):
        self.state = np.full((2, 6, 5), -1, dtype='int32')
        self.row_position = 0
        self.guess_num = 0
        if word == None:
            self.hidden_word = choice(tuple(self.hidden_words))
        else:
            self.hidden_word = tuple(('abcdefghijklmnopqrstuvwxyz'.index(c) for c in word))
        return (self.state, {})
    

    def step(self, action):

        self.state[0, self.guess_num, self.row_position] = action
        reward = 0
        terminal = False

        if self.row_position == 4:
            guess = tuple(self.state[0, self.guess_num])

            if guess == self.hidden_word:
                terminal = True
                reward = 7 - self.guess_num
                self.state[1, self.guess_num] = 2

            elif guess in self.allowed_words:
                for i in range(5):
                    if guess[i] == self.hidden_word[i]:
                        self.state[1, self.guess_num, i] = 2
                    elif guess[:i].count(guess[i]) + 1 <= self.hidden_word.count(guess[i]):
                        self.state[1, self.guess_num, i] = 1
                    else:
                        self.state[1, self.guess_num, i] = 0

                if self.guess_num == 5:
                    terminal = True
                    for letter in self.state[1, self.guess_num]:
                        if letter == 2:
                            reward += 0.2
                        elif letter == 1:
                            reward += 0.1
                        else:
                            reward -= 0.02
                
                else:
                    self.guess_num += 1

            else:
                self.state[0, self.guess_num] = -1
                reward = -0.1

            self.row_position = 0

        else:
            self.row_position += 1

        return (self.state, reward, terminal, False, {})


    def play(self, hidden_word = None):
        self.reset(hidden_word)
        terminal = False
        while True:
            display_string = '\n'
            for row in range(6):
                for position in range(5):
                    color = ('0', '33', '32','0')[self.state[1, row, position]]
                    letter = 'abcdefghijklmnopqrstuvwxyz.'[self.state[0, row, position]]
                    display_string += f'\033[{color}m{letter}'
                display_string += '\n'
            display_string += '\033[0m\n'
            if terminal:
                print(display_string)
                break
            word = input(display_string)
            word = tuple('abcdefghijklmnopqrstuvwxyz'.index(character) for character in word)
            rewards = []
            for character in word:
                ___, reward, terminal, _, __ = self.step(character)
                rewards.append(str(reward))
                if terminal:
                    break
            print(f'Rewards: ' + ' '.join(rewards))



def make():
    return Wordle_Environment()


if __name__ == '__main__':
    env = make()
    env.play()