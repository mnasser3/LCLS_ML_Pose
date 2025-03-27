import torch
import torch.nn as nn
from blocks.encoderDS import *
from blocks.unit_cell import *
from blocks.rotation_head import *
from blocks.diffraction_dataset import *
from blocks.encoderPN import *
from blocks.fourier_mapping import *
from blocks.encoderTransf import *
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
    
    def __init__(self, latent_dim=64, num_theta_samples=2, encoder_hidden=128, rotation_hidden=128,theta_isParam=False,theta_mu=None,theta_diagS=None,
                 use_fourier=True, fourier_mapping_size=16, fourier_scale=10.0,
                 set_transformer_heads=4, set_transformer_layers=2):
        super().__init__()
        self.use_fourier = use_fourier
        if self.use_fourier:
            self.fourier = FourierFeatureMapping(input_dim=3, mapping_size=fourier_mapping_size, scale=fourier_scale)
            input_dim = 3 + 2 * fourier_mapping_size
        else:
            input_dim = 3
        #self.encoder = EncoderPN(input_dim=input_dim, hidden_dim=encoder_hidden, output_dim=latent_dim)
        # self.encoder = EncoderDS(input_dim=input_dim, hidden_dim=encoder_hidden, output_dim=latent_dim)
        self.encoder = SetTransformerEncoder(input_dim=input_dim, embed_dim=encoder_hidden, 
                                        output_dim=latent_dim, num_heads=set_transformer_heads, 
                                        num_layers=set_transformer_layers)
        self.unit_cell = UnitCell(isParam=theta_isParam, num_samples=num_theta_samples, mu=theta_mu, diag_S=theta_diagS)  
        self.rotation_head = RotationHead(input_dim=latent_dim + 9, hidden_dim=rotation_hidden)
        self.norm = nn.LayerNorm(latent_dim)
    
    def forward(self, Q_batch,mask=None):
        """
        Args:
            Q_batch (torch.Tensor): A batch of diffraction matrices, shape [B, N, 3],
                                    where B is the batch size and N is the number of reciprocal vectors.
        
        Returns:
            R (torch.Tensor): Candidate rotation matrices, shape [B, C, 3, 3],
                              where C is the number of candidates from the UnitCell.
        """  
        if mask is None:
            mask = torch.ones(Q_batch.size(0), Q_batch.size(1), dtype=torch.bool, device=Q_batch.device)

        if self.use_fourier:
            fourier_feats = self.fourier(Q_batch)  # Shape: [B, N, 2*mapping_size]
            Q_input = torch.cat([Q_batch, fourier_feats], dim=-1)  # Now shape: [B, N, 3 + 2*mapping_size]
        else:
            Q_input = Q_batch

        # Q_batch shape: [B, N, 3] -> z shape: [B, latent_dim]
        b = Q_batch.shape[0]
        #z = self.enc