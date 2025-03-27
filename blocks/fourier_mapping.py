import torch
import torch.nn as nn
import math

class FourierFeatureMapping(nn.Module):
    def __init__(self, input_dim=3, mapping_size=16, scale=10.0):
        """
        Args:
            input_dim: Dimensionality of input Q (3 in our case).
            mapping_size: Number of frequencies (the output dimension will be 2 * mapping_size).
            scale: Scaling factor for the random frequencies.
        """
        super().__init__()
        self.F = nn.Parameter(torch.randn(input_dim, mapping_size) * scale, requires_grad=False)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, N, input_dim] (B: batch size, N: number of points).
        Returns:
            Tensor of shape [B, N, 2 * mapping_size] containing sine and cosine features.
        """
        x_proj = 2 * math.pi * torch.matmul(x, self.F)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
