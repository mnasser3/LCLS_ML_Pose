import gemmi
import numpy as np
import torch
from torch import nn, Tensor

class SymmetryAwareLossLoop(nn.Module):
    """
    Applies each rotation in `rot_mats` to the ground-truth orientation,
    then computes the geodesic loss to the prediction, taking the minimum.

    This version uses simple Python for-loops for clarity.
    """

    def __init__(self, base_loss: nn.Module, symm_group = "P 61 2 2"):
        """
        Args:
          base_loss: An instance of GeodesicLoss (or any rotation-based loss).
          rot_mats:  (S, 3, 3) Tensor of symmetry rotation matrices.
        """
        super().__init__()
        self.base_loss = base_loss
        rot_list = []
        sg = gemmi.SpaceGroup(symm_group)
        go = sg.operations()
        for op in go:
            R = torch.tensor(np.array(op.rot, dtype=float) / op.DEN, dtype=torch.float32)
            rot_list.append(R)

        rot_mats = torch.stack(rot_list, dim=0)
        self.register_buffer("rot_mats", rot_mats)

    def forward(self, R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
        """
        Vectorized version of the symmetry-aware loss function.
        R_pred: (B, C, 3, 3) - predicted rotation matrices (B=batch, C=candidates)
        R_gt: (B, 3, 3) - ground-truth rotation matrices
        Returns a scalar loss.
        """
        B, C, _, _ = R_pred.shape  # Batch size, number of candidates
        S = self.rot_mats.shape[0] # Number of symmetry operations

        # Expand dimensions to allow broadcasting
        R_gt_exp = R_gt[:, None, None, :, :]  # (B, 1, 1, 3, 3)
        R_pred_exp = R_pred[:, :, None, :, :]  # (B, C, 1, 3, 3)
        rot_mats_exp = self.rot_mats[None, None, :, :, :]  # (1, 1, S, 3, 3)

        # Apply all symmetry transformations to the ground-truth rotations
        R_gt_sym = torch.matmul(rot_mats_exp, R_gt_exp)  # (B, 1, S, 3, 3)

        # Flatten for batch processing in self.base_loss
        R_pred_flat = R_pred_exp.expand(-1, -1, S, -1, -1).reshape(B * C * S, 3, 3)  # (B*C*S, 3, 3)
        R_gt_sym_flat = R_gt_sym.expand(-1, C, -1, -1, -1).reshape(B * C * S, 3, 3)  # (B*C*S, 3, 3)

        # Compute the geodesic loss in a batch-wise manner
        all_losses = self.base_loss(R_pred_flat, R_gt_sym_flat)  # (B*C*S,)

        # Reshape back to (B, C, S)
        all_losses = all_losses.view(B, C, S)

        # Take the minimum loss over symmetry operations
        min_loss_sym, _ = torch.min(all_losses, dim=2)  # (B, C)

        # Take the minimum loss over candidates
        best_candidate_loss, _ = torch.min(min_loss_sym, dim=1)  # (B,)

        # Final mean over batch
        return best_candidate_loss.mean()

    
    def get_symmetry_rotations(self,space_group_name="P 61 2 2"):
        sg = gemmi.SpaceGroup("P 61 2 2")
        go = sg.operations()

        rot_mats = []
        for op in go:
            R = np.array(op.rot, dtype=float) / op.DEN  
            rot_mats.append(R)
        rot_mats = np.stack(rot_mats, axis=0)  
        return rot_mats









if __name__ == "__main__":

    sg = gemmi.SpaceGroup("P 61 2 2")
    go = sg.operations()
    for op in go:
        R = np.array(op.rot,dtype=float) / op.DEN
        print(R)

