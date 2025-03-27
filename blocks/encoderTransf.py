import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x, mask=None):
        # x: [B, N, embed_dim]
        # mask: [B, N] boolean; True for valid points.
        if mask is not None:
            # key_padding_mask expects True for positions that should be ignored.
            key_padding_mask = ~mask.bool()
        else:
            key_padding_mask = None
        # Self-attention: query, key, and value are all x.
        out, _ = self.mha(x, x, x, key_padding_mask=key_padding_mask)
        return out

class SetTransformerEncoder(nn.Module):
    def __init__(self, input_dim=3, embed_dim=128, output_dim=64, num_heads=4, num_layers=2):
        """
        Args:
            input_dim: Dimensionality of input per point (can be 3 or enriched with Fourier features).
            embed_dim: Hidden embedding dimension.
            output_dim: Final latent dimension.
            num_heads: Number of attention heads.
            num_layers: Number of self-attention layers.
        """
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.layers = nn.ModuleList([
            MultiHeadSelfAttention(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.PReLU(),
            nn.BatchNorm1d(embed_dim),
            nn.Linear(embed_dim, output_dim)
        )

    def forward(self, x, mask=None):
        # x: [B, N, input_dim]
        # mask: [B, N] (boolean) where True indicates a valid point.
        x = self.input_proj(x)  # [B, N, embed_dim]
        for layer in self.layers:
            residual = x
            x = layer(x, mask)
            x = x + residual  # residual connection
        # Pooling over the set using the mask if provided.
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()  # zero out padded entries
            summed = torch.sum(x, dim=1)
            count = mask.sum(dim=1, keepdim=True).clamp(min=1).float()
            pooled = summed / count
        else:
            pooled = torch.mean(x, dim=1)
        # Pass pooled feature through FC layers.
        z = self.fc(pooled)
        return z
