import torch
from torch import nn, Tensor

class FrobeniusLoss(nn.Module):
    """
    Computes Frobenius (MSE-style) loss between two rotation matrices.
    """
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, R_pred: Tensor, R_label: Tensor) -> Tensor:
        """
        R_pred: (N, 3, 3)
        R_label: (N, 3, 3)
        """
        diff = R_pred - R_label
        loss = torch.norm(diff, dim=(1, 2)) ** 2  # Frobenius norm squared

        if self.reduction == "none":
            return loss  # shape: (N,)
        elif self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError(f"Unsupported reduction mode: {self.reduction}")
