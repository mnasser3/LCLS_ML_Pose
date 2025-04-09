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

    def __init__(self, base_loss: nn.Module, cell=torch.tensor([93.1636, 93.1636, 130.6054, 90.0, 90.0, 120.0])):
        """
        Args:
          base_loss: An instance of GeodesicLoss (or any rotation-based loss).
          rot_mats:  (S, 3, 3) Tensor of symmetry rotation matrices.
        """
        super().__init__()
        self.base_loss = base_loss
        rot_list = []
        
        # Get symmetry operations in fractional space
        cell = gemmi.UnitCell(*cell)
        ops = gemmi.find_lattice_symmetry(cell, 'P', 3.0)
        B=(np.array(cell.frac.mat).T)
        B_inv = np.array(cell.orth.mat).T

        for i, op in enumerate(ops):
            R_frac = np.array(op.rot, dtype=float) / op.DEN
            R_recip = B @ R_frac.T @ B_inv
            R = torch.tensor(R_recip, dtype=torch.float32)
            rot_list.append(R)
        rot_mats = torch.stack(rot_list, dim=0) 

        # Check if all matrices in rot_mats are valid SO(3) matrices
        def is_so3(matrix, atol=1e-1):
            identity = torch.eye(3, device=matrix.device)
            RRT = torch.matmul(matrix, matrix.transpose(-1, -2))
            ortho = torch.allclose(RRT, identity, atol=atol)
            det = torch.det(matrix)
            det_close = torch.isclose(det, torch.tensor(1.0, device=matrix.device), atol=atol)
            return ortho and det_close

        if not all(is_so3(matrix) for matrix in rot_mats.view(-1, 3, 3)):
            for i, matrix in enumerate(rot_mats.view(-1, 3, 3)):
                identity = torch.eye(3, device=matrix.device)
                RRT = matrix @ matrix.transpose(-1, -2)
                diff = RRT - identity
                ortho_error = torch.norm(diff)
                det = torch.det(matrix).item()

                if not is_so3(matrix):
                    print(f"Matrix {i} failed SO(3) check.")
                    print(f"  Determinant     = {det:.6f}")
                    print(f"  Orthogonality error (||RᵀR - I||) = {ortho_error:.6e}")
                    if np.isclose(det, -1.0, atol=1e-3):
                        print("  -> Determinant is approximately -1: improper rotation")
            raise ValueError("rot_mats contains matrices that are not valid SO(3) rotations.")

        self.register_buffer("rot_mats", rot_mats)

    def forward(self, R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
        """
        Vectorized version of the symmetry-aware loss function.
        R_pred: (B, C, 3, 3) - predicted rotation matrices (B=batch, C=candidates)
        R_gt: (B, 3, 3) - ground-truth rotation matrices
        Returns a scalar loss.
        """
        B, C, _, _ = R_pred.shape
        S = self.rot_mats.shape[0]
        
        def is_so3(matrix, atol=1e-1):
            identity = torch.eye(3, device=matrix.device)
            RRT = torch.matmul(matrix, matrix.transpose(-1, -2))
            ortho = torch.allclose(RRT, identity, atol=atol)
            det = torch.det(matrix)
            det_close = torch.isclose(det, torch.tensor(1.0, device=matrix.device), atol=atol)
            return ortho and det_close
        
        R_gt_exp = R_gt[:, None, None, :, :]
        R_pred_exp = R_pred[:, :, None, :, :]
        rot_mats_exp = self.rot_mats[None, None, :, :, :].to(R_pred.device)

        R_gt_sym = torch.matmul(R_gt_exp, rot_mats_exp)

        R_pred_flat = R_pred_exp.expand(-1, -1, S, -1, -1).reshape(B * C * S, 3, 3)
        R_gt_sym_flat = R_gt_sym.expand(-1, C, -1, -1, -1).reshape(B * C * S, 3, 3)
        
        # Check if SO(3) matrices
        # if not all(is_so3(matrix) for matrix in R_pred_flat.reshape(-1, 3, 3)):
        #     raise ValueError("R_pred_exp contains matrices that are not valid SO(3) rotations.")
        # if not all(is_so3(matrix) for matrix in R_gt_sym_flat.reshape(-1, 3, 3)):
        #     raise ValueError("R_gt_sym contains matrices that are not valid SO(3) rotations.")

        all_losses = self.base_loss(R_pred_flat, R_gt_sym_flat)

        all_losses = all_losses.view(B, C, S)

        avg_loss_over_C = all_losses.mean(dim=1)  # average over C
        min_loss_sym, _ = torch.min(avg_loss_over_C, dim=1)  # min over S
        return min_loss_sym.mean()
