from random import randint, random

class Wordle_Row_Environment:

    def __init__(self):
        self.allowed_words = set()
        with open("data/dictionary.txt", 'r') as file:
            for line in file:
                word = tuple('abcdefghijklmnopqrstuvwxyz'.index(character) for character in line.strip().lower())
                self.allowed_words.add(word)


    def reset(self):
        self.state = (26, 26, 26, 26, 26)
        self.position = 0
        return self.state
    

    def step(self, action):

        if action == 26:
            if self.position == 5 and self.state in self.allowed_words:
                return ((27, 27, 27, 27, 27), 100, True)
            else:
                return (self.state, -10, False)
            
        if action == 27:
            if self.position != 0:
                self.position -= 1
                self.state = self.state[0:self.position] + (26,) * (5 - self.position)
            return (self.state, -10, False)
            
        if self.position != 5:
            self.state = self.state[0:self.position] + (action,) + self.state[self.position + 1:]
            self.position += 1
        return (self.state, -1, False)
    


def test_simple():
    env = Wordle_Row_Environment()
    env.reset()
    env.step(0)
    env.step(1)
    env.step(1)
    env.step(4)
    env.step(24)
    (state, reward, terminal) = env.step(26)
    print(reward)