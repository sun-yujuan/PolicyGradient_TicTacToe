from typing import Tuple, List
import numpy as np
import torch
from torch import nn
from env.Environment import Environment


# Utility Functions
def index_to_coord(index: int, side_length: int) -> Tuple[int, int]:
    """
    Converts a linear index to row and column coordinates on a grid.

    Args:
        index: Linear index of a cell on the grid.
        side_length: Side length of the grid.

    Returns:
        Tuple[int, int]: Row and column coordinates corresponding to the index.
    """
    row, col = divmod(index, side_length)
    return row, col


def compute_returns(rewards: List[float], gamma: float = 1.0) -> List[float]:
    """
    Computes the cumulative returns for each time step, given rewards and discount factor.

    Args:
        rewards: List of rewards obtained at each time step.
        gamma: Discount factor for future rewards.

    Returns:
        List[float]: Cumulative returns for each time step.
    """
    returns = np.zeros_like(rewards, dtype=float)
    returns[-1] = rewards[-1]
    for t in range(len(returns) - 2, -1, -1):
        returns[t] = rewards[t] + gamma * returns[t + 1]
    return returns


# Agent Class
class Agent:
    """
    Represents an agent interacting with a game environment. The agent can use a neural network
    to decide actions and learn from experiences. If no network is provided, the agent acts randomly.

    Attributes:
        env (Environment): The game environment.
        net (nn.Module): Neural network for action decisions.
        lr (float): Learning rate for optimization.
        weight_decay (float): L2 regularization coefficient.
    """

    def __init__(self, env: Environment, policy_net: nn.Module = None, lr=0.002, weight_decay=0.01) -> None:
        self.env = env
        self.policy_net = policy_net
        self.reset()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if policy_net:
            self.policy_net = self.policy_net.to(self.device)
            self.policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr, weight_decay=weight_decay)

        # Training attributes
        self.saved_rewards = []
        self.saved_log_probs = []
        self.saved_state_values = []
        self.invalid_move_count = 0

    def reset(self) -> None:
        """Resets the environment and clears training attributes."""
        self.saved_rewards = []
        self.saved_log_probs = []
        self.saved_state_values = []
        self.invalid_move_count = 0

    def get_action_random(self) -> Tuple[int, int]:
        """Returns a random action using the environment's method."""
        return self.env.get_computer_move()

    def get_action(self, is_eval: bool) -> Tuple[int, int]:
        """Gets an action from the agent, depending on the evaluation or training mode."""
        pass

    def make_move(self, is_eval: bool) -> bool:
        """
        Makes a move in the environment, either randomly or using the policy network.

        Args:
            is_eval: Indicates if the agent is in evaluation or training mode.

        Returns:
            bool: True if the game has ended after the agent's move, False otherwise.
        """
        if self.policy_net is None:
            reward, done = self.env.step(self.get_action_random())
        else:
            reward, done = self.env.step(self.get_action(is_eval))
            self.saved_rewards.append(reward)

        return done

    def update_network(self, gamma: float) -> float:
        """
        Updates the agent's neural network using the saved rewards and the discount factor.

        Args:
            gamma: Discount factor for future rewards.

        Returns:
            float: Total loss in the episode.
        """
        pass
