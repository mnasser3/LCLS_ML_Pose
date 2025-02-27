import torch
import torch.nn as nn
from blocks.encoder import *
from blocks.unit_cell import *
from blocks.rotation_head import *
from blocks.diffraction_dataset import *
import numpy as np
from blocks.assign_head import AssignHead

class QtoHModel(nn.Module):
    """
    QtoH Model: Predicts Miller indices h from diffraction data Q,
    and the outputs from QtoR (R, B, z).
    
    Given:
        - Q: Diffraction matrix of shape [B, N, 3]
        - R: Candidate rotation matrices from QtoR, shape [B, C, 3, 3]
        - B: Candidate reciprocal basis matrices, shape [C, 3, 3]
        - z: Latent representation from the encoder, shape [B, latent_dim]
    

    concatenates q,B^-T,R^-1 with z (latent encoder of Q), and then uses the Assign Head
    to predict the Miller indices.
    
    Output:
        h: Predicted Miller indices, shape [B, C, N, 3]
    """
    
    def __init__(self, assign_hidden=128, N_max=10, temperature=1.0):
        super().__init__()
        self.assign_head = AssignHead(hidden_dim=assign_hidden, N_max=N_max, temperature=temperature)
    
    def forward(self, Q, R, B, z):
        """
        Args:
            Q (torch.Tensor): Diffraction data, shape [B, N, 3]
            R (torch.Tensor): Candidate rotations, shape [B, C, 3, 3]
            B (torch.Tensor): Candidate reciprocal basis matrices, shape [C, 3, 3]
            z (torch.Tensor): Latent representation, shape [B, latent_dim]
        
        Returns:
            h (torch.Tensor): Predicted Miller indices, shape [B, C, N, 3]
        """
        b = Q.shape[0]  # B
        c = B.shape[0]  # C
        N = Q.shape[1]  # N
        
        B_inv = torch.inverse(B)  # [C, 3, 3]
        B_inv = B_inv.unsqueeze(0).expand(b,c,3,3)  # Shape: [B, C, 3, 3]
        R_inv = R.mT  # [B, C, 3, 3]
        M = torch.matmul(B_inv,R_inv)
        
        B_inv_flat = B_inv.reshape(b, c, -1) # [B, C, 9]
        R_inv_flat = R_inv.reshape(b, c, -1)  # [B, C, 9]
        M_flat = M.reshape(b, c, -1)
        
        #NOTE: Decide which features to use for the AssignHead 
        B_inv_exp = B_inv_flat.unsqueeze(2).expand(b,c, N, 9) # [B, C, N, 9]
        R_inv_exp = R_inv_flat.unsqueeze(2).expand(b, c, N, 9) # [B, C, N, 9]
        M_exp = M_flat.unsqueeze(2).expand(b, c, N, M_flat.shape[-1]) # [B, C, N, 9]
        Q_exp = Q.unsqueeze(1).expand(b, c, N, 3)  # [B, C, N, 3]
        z_exp = z.unsqueeze(1).unsqueeze(2).expand(b,c, N, z.shape[-1])  # [B, C, N, latent_dim]
        
        #NOTE: Decide on the lambda value
        lambda_ = 0.5
        features = torch.cat([Q_exp,lambda_*M_exp,z_exp], dim=-1)
        
        features_flat = features.view(b,c, N, features.shape[-1])
        h_flat = self.assign_head(features_flat)
        h = h_flat.view(b,c, N, 3)
           
        return h