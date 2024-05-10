from wordle_environment import make
import json



def get_colours(word, word_counts, enum_guess, start_counts):
    result = [0, 0, 0]
    reward = 0 #(0, 0.05, 0.1)
    guess_counts = start_counts.copy()
    for i, l in enum_guess:
        if l == word[i]:
            result[i] = 2
            guess_counts[l] += 1
            reward += 0.1
    for i, l in enum_guess:
        if l != word[i]:
            try:
                if guess_counts[l] < word_counts[l]:
                    guess_counts[l] += 1
                    result[i] = 1
                    reward += 0.05
            except KeyError:
                pass
    return tuple(result), reward



def solve(vocab, hidden_words, guesses_left, step_rewards):
    if guesses_left == 0:
        return (hidden_words[0][0], {}), step_rewards
    if len(hidden_words) == 1:
        return (hidden_words[0][0], {}), 10 + guesses_left * 0.2 + step_rewards - 0.0003
    
    hidden_words_len = len(hidden_words)
    
    best_avg_reward = None
    for guess in vocab:
        solution_candidate = {}
        avg_reward = 0

        results = {}
        new_vocab = vocab.copy()
        new_vocab.remove(guess)

        enumerated_guess = tuple(enumerate(guess))
        starting_guess_counts = {l:0 for l in guess}
        for word, word_counts in hidden_words:
            result, reward = get_colours(word, word_counts, enumerated_guess, starting_guess_counts)
            try:
                results[result][0].append((word, word_counts))
            except KeyError:
                results[result] = ([(word, word_counts)], reward)

        if len(results) != 1:

            i = 0
            for result in results:
                if guesses_left == 6:
                    i += 1
                    print(f'guess {vocab.index(guess)+1} of {len(vocab)}, result {i} of {len(results)}')
                words, final_guess_reward = results[result]
                if result == (2, 2, 2):
                    avg_reward += 10 + guesses_left * 0.2 + step_rewards - 0.0003
                else:
                    mini_solution, mini_avg_reward = solve(new_vocab, words, guesses_left - 1, step_rewards-0.0003)
                    solution_candidate[result] = mini_solution
                    if guesses_left == 1:
                        mini_avg_reward += final_guess_reward
                    avg_reward += mini_avg_reward * len(words)
            
            if best_avg_reward == None or avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                solution = (guess, solution_candidate)

    return solution, best_avg_reward / hidden_words_len



def encode_keys(solutions_branch):
    guess, results_dict = solutions_branch
    new_results_dict = {}
    for result in results_dict:
        encoded_key = 0
        for i, l in enumerate(result):
            encoded_key += l << (5 * i)
        new_results_dict[encoded_key] = encode_keys(results_dict[result])
    return guess, new_results_dict



def decode_keys(solutions_branch):
    guess, results_dict = solutions_branch
    new_results_dict = {}
    for result in results_dict:
        decoded_key = tuple((int(result) >> (5 * i)) & (2 ** 5 - 1) for i in range(3))
        new_results_dict[decoded_key] = decode_keys(results_dict[result])
    return guess, new_results_dict



def make_solutions_tree(env):
    vocab = list(env.vocab_tuples)
    hidden_words = [(word[1], word[4]) for word in env.hidden_words]
    solutions = solve(vocab, hidden_words, env.max_guesses, 0)[0]
    with open('optimal_guesses.json', 'w') as f:
        json.dump(encode_keys(solutions), f)
    return solutions



settings = {
    'word_length': 3,
    'truncation_limit': 1000,
    'correct_guess_reward': 10,
    'early_guess_reward': 0.2,
    'colour_rewards': (0, 0.05, 0.1),
    'valid_word_reward': 0,
    'invalid_word_reward': 0,
    'step_reward': -0.0001,
    'repeated_guess_reward': 0,
    'alphabet': 'abcdefgh',
    'vocab_file': 'word_lists/three_letter_abcdefgh.txt',
    'hidden_words_file': 'word_lists/three_letter_abcdefgh.txt',
    'state_representation': 'int',
    'max_guesses':6
}
env = make(settings)
try:
    with open('optimal_guesses.json', 'r') as f:
        solutions = json.load(f)
        solutions = decode_keys(solutions)
except FileNotFoundError:
    solutions = make_solutions_tree(env)

total_reward = 0
wins = 0
guesses = 0
for word, *_ in env.hidden_words:
    state, _ = env.reset(word)
    solutions_branch = solutions
    for i in range(6):
        guess = solutions_branch[0]
        for action in guess:
            state, reward, terminal, truncated, info = env.step(action)
        guesses += 1
        if terminal:
            total_reward += info['total_reward']
            if info['correct_guess']:
                wins += 1
            break
        solutions_branch = solutions_branch[1][tuple(state[1, i])]

reward = total_reward / len(env.hidden_words)
winrate = wins / len(env.hidden_words)
guesses /= len(env.hidden_words)
print(f'Avg reward: {reward} \t Winrate: {winrate} \t Avg guesses: {guesses}')