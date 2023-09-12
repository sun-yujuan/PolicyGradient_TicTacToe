from typing import Tuple
from torch import nn
import torch
from network.ResidualBlock import ResidualBlock


class PolicyWithValueNetwork(nn.Module):
    """
    This class implements a policy network with a baseline (value network).
    It uses a neural network with residual blocks and dropout for regularization.
    The output of this network consists of a probability distribution over the possible actions
    and an estimate of the value of the current state (baseline).
    """

    def __init__(self, num_cells: int, hidden_features: int, num_layers: int, dropout_probability: float,
                 num_output: int) -> None:
        """
        Initialize a new instance of the policy network with baseline.

        Args:
            num_cells (int): The number of cells in the board.
            hidden_features (int): The number of features in the hidden layer of the network.
            num_layers (int): The number of layers in the network.
            dropout_probability (float): The probability of dropout for regularization.
        """
        super().__init__()

        input_layer = [
            nn.Linear(in_features=num_cells * 3, out_features=hidden_features),  # Linear layer
            nn.ReLU(),  # Apply ReLU activation function
            nn.Dropout(p=dropout_probability),  # Apply Dropout for regularization
        ]

        hidden_layers = [
            ResidualBlock(in_features=hidden_features, out_features=hidden_features,
                          dropout_probability=dropout_probability)
            for _ in range(num_layers)  # Create specified number of hidden layers
        ]

        self.model = nn.Sequential(*input_layer, *hidden_layers)  # Combine input layer and hidden layers

        # The policy outputs a probability distribution over the actions
        self.policy_network = nn.Sequential(
            nn.Linear(in_features=hidden_features, out_features=num_output),
            nn.Softmax(dim=-1)  # Apply Softmax to ensure these probabilities sum to 1
        )

        # The value network outputs an estimate of the state's value
        self.value_network = nn.Linear(hidden_features, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The current state of the game, which is the one-hot encoding of the game board .

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The action probabilities and state value.
        """
        x1 = self.model(x)  # Apply the main network to the input
        action_probs = self.policy_network(x1)  # Apply the actor network to get the action probabilities

        x2 = self.model(x)  # Apply the main network to the input again
        state_value = self.value_network(x2)  # Apply the critic network to get the state value

        return action_probs, state_value
