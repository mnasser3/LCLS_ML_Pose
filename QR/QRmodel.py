import torch
import torch.nn as nn
from blocks.encoder import *
from blocks.unit_cell import *
from blocks.rotation_head import *
from blocks.diffraction_dataset import *
import numpy as np


class QtoRModel(nn.Module):
    """
    A test model that processes a batch of diffraction matrices Q_i:
      1. Uses a DeepSets encoder to get latent representation z (shape: [B, latent_dim]).
      2. Uses the UnitCell module to get multiple candidate reciprocal basis matrices B (shape: [C, 3, 3]).
      3. For each candidate, concatenates a flattened B (shape: [9]) to z,
         and passes the result through the RotationHead to produce candidate rotations R.
         
    Output R has shape: [B, C, 3, 3].
    """
    
    def __init__(self, latent_dim=64, num_theta_samples=2, encoder_hidden=128, rotation_hidden=128,theta_isParam=False,theta_mu=None,theta_diagS=None):
        super().__init__()
        self.encoder = Encoder(input_dim=3, hidden_dim=encoder_hidden, output_dim=latent_dim)
        self.unit_cell = UnitCell(isParam=theta_isParam, num_samples=num_theta_samples, mu=theta_mu, diag_S=theta_diagS)  
        # Our rotation head now expects input dimension = latent_dim + 9 (flattened B candidate)
        self.rotation_head = RotationHead(input_dim=latent_dim + 9, hidden_dim=rotation_hidden)
    
    def forward(self, Q_batch,mask):
        """
        Args:
            Q_batch (torch.Tensor): A batch of diffraction matrices, shape [B, N, 3],
                                    where B is the batch size and N is the number of reciprocal vectors.
        
        Returns:
            R (torch.Tensor): Candidate rotation matrices, shape [B, C, 3, 3],
                              where C is the number of candidates from the UnitCell.
        """  
        # Q_batch shape: [B, N, 3] -> z shape: [B, latent_dim]
        b = Q_batch.shape[0]
        z = self.encoder(Q_batch,mask)
        
        # unit_cell.forward() returns B_candidate of shape [C, 3, 3] -> flatten to [C, 9]
        B_candidates, _, _ = self.unit_cell()  
        c = B_candidates.shape[0]
        B_flat = B_candidates.reshape(c, -1) 
        
        # Repeat z for each candidate B_i: shape becomes [B, C, latent_dim]
        z_repeated = z.unsqueeze(1).expand(b, c, z.shape[-1])
        B_flat_repeated = B_flat.unsqueeze(0).expand(b, c, 9)
        zB = torch.cat([z_repeated, B_flat_repeated], dim=2)
        
        # The rotation head expects input shape [B, C, input_dim] and outputs R of shape [B, C, 3, 3]
        R = self.rotation_head(zB)
        return R

            



