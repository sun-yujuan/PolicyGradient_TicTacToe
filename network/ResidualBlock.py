import torch
from torch import nn


class ResidualBlock(nn.Module):
    """
    A residual block takes an input, applies transformations, and adds the transformed input back to the original
    input. This helps in mitigating the problem of vanishing gradients and allows the network to learn identity
    functions more easily, which can lead to improved performance.
    """

    def __init__(self, in_features: int, out_features: int, dropout_probability: float) -> None:
        """Initializes residual MLP-block."""
        super().__init__()

        self.mlp_block = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.ReLU(),
            nn.Linear(in_features=out_features, out_features=out_features),

            # Dropout is applied, it randomly zeros some of the elements of the input tensor with probability p
            nn.Dropout(p=dropout_probability),
            nn.LayerNorm(out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass applies the MLP block to the input, then adds the original input to the result
        # This is the "residual connection" part
        return x + self.mlp_block(x)
