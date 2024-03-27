# woRdLe
Solving Wordle with reinforcement learning. For University of Bath unit CM50270

Requires NumPy

I've tried to follow the gym api as much as possible so check that if you have any issues, but the long and the short of it is:
- Action can be any number from 0 to 25 (0 is a, 1 is b etc.)
- State is a 3d array. 1st dim: 0 = letters, 1 = colours. 2nd dim is rows, 3rd dim is position along the row.
- Colours are represented as follows: 0 = grey, 1 = yellow, 2 = green
- A value of -1 in either array means that cell hasn't been filled in yet

There is a .play() method which lets you play in the command line and see colours + rewards - you can also manually set the hidden word by passing it as an argument, e.g. env.play('abbey')