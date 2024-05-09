import numpy as np
import matplotlib.pyplot as plt
import json



def average_per_n(data, n):
    return [np.mean(data[i:i+n]) for i in range(0, len(data), n)]



def plot_rewards(agent_data, human_datum, optimal_datum, filename, min_x = None):
    plt.clf()
    plt.axline(
        [0, optimal_datum],
        [1, optimal_datum],
        c=optimal_colour,
        linewidth=optimal_linewidth,
        label = optimal_label,
        linestyle = 'dashed'
    )
    plt.axline(
        [0, human_datum],
        [1, human_datum],
        c=human_colour,
        linewidth=human_linewidth,
        label = human_label,
        linestyle = 'dotted'
    )
    plt.plot(
        agent_data,
        c=training_colour,
        linewidth=training_linewidth,
        label = training_label
    )
    if min_x != None:
        max_x = len(agent_data)
        offset = (max_x - min_x) * 33 / 678
        plt.xlim(min_x - offset, max_x + offset)
        data_slice = agent_data[min_x:]
        fixed_points = [human_datum, optimal_datum]
        min_y, max_y = min(data_slice), max(data_slice)
        min_y, max_y = min([min_y] + fixed_points), max([max_y] + fixed_points)
        offset = (max_y - min_y) * 33 / 678
        plt.ylim(min_y - offset, max_y + offset)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.minorticks_on()
    plt.grid(True, which='minor', c='gainsboro', linewidth=0.5)
    plt.grid(True, which='major')
    plt.legend()
    plt.savefig('graphs/' + filename + '.svg', format="svg")



x_label = 'x100 episodes'
y_label = 'Average total episode reward'

explore_y_label = 'Average total episode exploration reward'

training_label = 'Agent'
human_label = 'Human'
optimal_label = 'Optimal'

training_colour = 'black'
human_colour = 'blue'
optimal_colour = 'steelblue'

training_linewidth = 0.7
human_linewidth = 2
optimal_linewidth = 2

n = 100
training_rewards = np.load('final_model/episode_rewards.npy')
avg_training_rewards = average_per_n(training_rewards, n)
training_winrate = [int(reward > 0.3) for reward in training_rewards]
avg_training_winrate = average_per_n(training_winrate, n)

avg_optimal_reward = 10.4
avg_optimal_winrate = 1

human_data = []
for filename in ('ben.json', 'james.json', 'leena.json', 'veronika.json', 'raps.json'):
    with open('human_data/' + filename, 'r') as f:
        human_data += json.load(f)['episodes']
human_rewards = [episode['total_reward'] for episode in human_data]
avg_human_reward = np.mean(human_rewards)
avg_human_winrate = np.mean([int(episode['info']['correct_guess']) for episode in human_data])

avg_explore_rewards = average_per_n(np.load('final_model/episode_explore_rewards.npy'), n)
plt.clf()
plt.plot(avg_explore_rewards, c=training_colour, linewidth=training_linewidth)
plt.xlabel(x_label)
plt.ylabel(explore_y_label)
plt.minorticks_on()
plt.grid(True, which='minor', c='gainsboro', linewidth=0.5)
plt.grid(True, which='major')
plt.savefig('graphs/explore.svg', format="svg")

plot_rewards(avg_training_rewards, avg_human_reward, avg_optimal_reward, 'rewards')
y_label = 'Winrate'
plot_rewards(avg_training_winrate, avg_human_winrate, avg_optimal_winrate, 'winrate')
x_label = 'Episode'
y_label = 'Total episode reward'
plot_rewards(training_rewards, avg_human_reward, avg_optimal_reward, 'final_rewards', 79800)

