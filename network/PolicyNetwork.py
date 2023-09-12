from torch import nn
import torch
from network.ResidualBlock import ResidualBlock


class PolicyNetwork(nn.Module):
    """
    This class implements the policy network that's used to decide the agent's actions.
    It uses a neural network with residual blocks and dropout for regularization.
    The output of this network is a probability distribution over the possible actions.
    """

    def __init__(self, num_cells: int, hidden_features: int, num_layers: int, dropout_probability: float,
                 num_output: int) -> None:
        """
        Initialize a new instance of the policy network.

        Args:
            num_cells (int): The number of cells in the board.
            hidden_features (int): The number of features in the hidden layer of the network.
            num_layers (int): The number of layers in the network.
            dropout_probability (float): The probability of dropout for regularization.
        """
        super().__init__()

        input_layer = [
            # Since the usage of one-hot encoding the in_features is 3 times the number of cells in the board
            # ('X', 'O' and empty)
            nn.Linear(in_features=num_cells * 3, out_features=hidden_features),
            nn.ReLU(),  # Apply ReLU activation function
            nn.Dropout(p=dropout_probability),  # Apply Dropout for regularization
        ]

        hidden_layers = [
            ResidualBlock(in_features=hidden_features, out_features=hidden_features,
                          dropout_probability=dropout_probability)
            for _ in range(num_layers)  # Create specified number of hidden layers
        ]

        self.model = nn.Sequential(*input_layer, *hidden_layers)  # Combine input layer and hidden layers

        # The actor outputs a probability distribution over the actions
        self.actor = nn.Sequential(
            nn.Linear(in_features=hidden_features, out_features=num_output),
            nn.Softmax(dim=-1)  # Apply Softmax to ensure these probabilities sum to 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The current state of the game, which is the one-hot encoding of the game board .

        Returns:
            torch.Tensor: The action probabilities.
        """
        x = self.model(x)  # Apply the main network to the input
        action_probs = self.actor(x)  # Apply the actor network to get the action probabilities

        return action_probs
