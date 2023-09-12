# Policy Gradient for Tic Tac Toe

This project contains implementations of the Reinforce, Reinforce with Baseline, and Actor-Critic algorithms used to train agents to play the game of Tic Tac Toe.

## Project Structure

The project is organized as follows:

- **[agents_tictactoe](agents_tictactoe)**: This directory contains the definition of the Agent class that represents the game-playing agent. The agent utilizes the Reinforce, Reinforce with Baseline, and Actor-Critic algorithms to learn to play the game.
- **[train.py](train.py)**: This file contains the training loop for the agent. It includes functions for running training episodes, training the agents, testing the agents, and analyzing the model.
- **[play_with_human.py](play_with_human.py)**: This file contains a function for playing a game against the trained agent.
- **[reinforce_tictactoe.ipynb](reinforce_tictactoe.ipynb) or other notebooks**: This Jupyter notebook contains code for training the reinforce agent to play Tic Tac Toe.
- **[env](env)**: This directory contains the environment class for the Tic Tac Toe game.
- **[models](models)**: This directory contains saved models of the trained agents.
- **[network](network)**: This directory contains the code for the policy network, reinforce with baseline network, and the actor-critic network.
- **[runs](runs)**: This directory contains output logs from TensorBoard.

## Usage

First, train the agent using the [reinforce_tictactoe.ipynb](reinforce_tictactoe.ipynb) or other notebook. This will generate a trained model in the [models](models) directory.

Then, you can play against the trained agent using the [play_with_human.py](play_with_human.py) script. To do this, run the script and follow the prompts to play a game.

To visualize the training process, you can use TensorBoard by typing the command `tensorboard --logdir=runs/tictactoe_200k(or 80k/400k)` in the terminal. The logs are located in the `runs/` directory.

## Dependencies

- Python 3.10
- PyTorch
- TensorBoard
- numpy
