import torch
from torch import nn, Tensor

class LieAlgebraLoss(nn.Module):
    """
    Computes the Lie algebra-based loss for SO(3) rotations.
    Instead of minimizing geodesic distance directly, it minimizes the norm
    of the logarithm map (which represents the minimal rotation vector).
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-6) -> None:
        super().__init__()
        self.reduction = reduction
        self.eps = eps  # Small constant to improve numerical stability

    def so3_log(self, R: Tensor) -> Tensor:
        """
        Compute the logarithm map from SO(3) to so(3) (Lie algebra).
        Input: R (batch_size, 3, 3) rotation matrix
        Output: omega (batch_size, 3) Lie algebra vector
        """
        trace = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)
        theta = torch.acos(torch.clamp((trace - 1) / 2, -1.0 + self.eps, 1.0 - self.eps)) 
        
        small_angle = theta < self.eps
        large_angle = ~small_angle 
        
        omega_hat = (R - R.transpose(-2, -1)) / (2 * torch.sin(theta).unsqueeze(-1).unsqueeze(-1) + self.eps)
        omega = torch.stack([omega_hat[..., 2, 1], omega_hat[..., 0, 2], omega_hat[..., 1, 0]], dim=-1)
        
        omega[small_angle] = 0.5 * torch.stack([
            R[small_angle, 2, 1] - R[small_angle, 1, 2],
            R[small_angle, 0, 2] - R[small_angle, 2, 0],
            R[small_angle, 1, 0] - R[small_angle, 0, 1]
        ], dim=-1)
        
        omega[large_angle] = theta[large_angle].unsqueeze(-1) * omega[large_angle]
        
        return omega

    def forward(self, R_pred: Tensor, R_label: Tensor) -> Tensor:
        """
        Compute the Lie algebra loss.
        R_pred: Predicted SO(3) rotation matrix (batch_size, 3, 3)
        R_label: Ground truth SO(3) rotation matrix (batch_size, 3, 3)
        """
        R_diff = R_label.transpose(-2, -1) @ R_pred  # Compute relative rotation
        omega = self.so3_log(R_diff)  # Convert to Lie algebra
        loss = torch.norm(omega, dim=-1)  # L2 norm of the rotation vector

        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
