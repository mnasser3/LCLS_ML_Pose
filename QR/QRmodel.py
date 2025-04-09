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
        self.encoder = EncoderDS(input_dim=input_dim, hidden_dim=encoder_hidden, output_dim=latent_dim)
        #self.encoder = SetTransformerEncoder(input_dim=input_dim, embed_dim=encoder_hidden, output_dim=latent_dim, num_heads=set_transformer_heads,num_layers=set_transformer_layers)
        self.unit_cell = UnitCell(isParam=theta_isParam, num_samples=num_theta_samples, mu=theta_mu, diag_S=theta_diagS)  
        self.rotation_head = RotationHead(input_dim=latent_dim + 20, hidden_dim=rotation_hidden)
        self.norm = nn.LayerNorm(latent_dim+20)
        self.norm2 = nn.LayerNorm(9)
        self.dropout = nn.Dropout(p=0.3)
    
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


        # For Transformer + DS
        if self.use_fourier:
            fourier_feats = self.fourier(Q_batch)  # Shape: [B, N, 2*mapping_size]
            Q_input = torch.cat([Q_batch, fourier_feats], dim=-1)  # Now shape: [B, N, 3 + 2*mapping_size]
        else:
            Q_input = Q_batch
        
        # # For PN 
        # if self.use_fourier:
        #     extra_features = self.fourier(Q_batch)  # Shape: [B, N, 2*mapping_size]
        # else:
        #     extra_features = None

        # Q_batch shape: [B, N, 3] -> z shape: [B, latent_dim]
        b = Q_batch.shape[0]
        n = Q_batch.shape[1]
        
        # for PN
        #z = self.encoder(Q_batch,mask,features=extra_features) 
        
        # for DS and Transformer
        z = self.encoder(Q_input, mask) #for DS and trans
        #z = self.dropout(z)
        
        
        
        Q_mean = Q_batch.mean(dim=1, keepdim=True)              # [B, 1, 3]
        Q_centered = Q_batch - Q_mean                           # [B, N, 3]
        cov = torch.bmm(Q_centered.transpose(1, 2), Q_centered)   # [B, 3, 3]
        cov = cov / (n - 1)                                     # [B, 3, 3]
        cov_flat = cov.view(b, -1)                              # [B, 9]

        Q_norm = F.normalize(Q_batch, dim=-1)  # [B, N, 3]
        pairwise_dot = torch.bmm(Q_norm, Q_norm.transpose(1, 2))  # [B, N, N]
        eye_mask = 1.0 - torch.eye(n, device=Q_batch.device, dtype=Q_batch.dtype).unsqueeze(0)  # [1, N, N]
        pairwise_vals = pairwise_dot * eye_mask  # [B, N, N]
        num_offdiag = n * (n - 1)
        pairwise_mean = pairwise_vals.sum(dim=(1, 2)) / num_offdiag  # [B]
        diff = (pairwise_vals - pairwise_mean.view(b, 1, 1)) * eye_mask
        pairwise_std = torch.sqrt((diff ** 2).sum(dim=(1, 2)) / num_offdiag)  # [B]
        pairwise_summary = torch.stack([pairwise_mean, pairwise_std], dim=-1)  # [B, 2]

        # Concatenate the two types of extra features: cov_flat (9-dim) and pairwise_summary (2-dim)
        extra_features = torch.cat([cov_flat, pairwise_summary], dim=-1)   # [B, 11]
        z = torch.cat([z, extra_features], dim=-1)  # [B, latent_dim + 11]
        
        
        
        # unit_cell.forward() returns B_candidate of shape [C, 3, 3] -> flatten to [C, 9]
        B_candidates, _, _ = self.unit_cell()  
        c = B_candidates.shape[0]
        B_flat = B_candidates.reshape(c, -1) 
        # print("Mean and std of B_flat:", B_flat.mean(dim=0), B_flat.std(dim=0,unbiased=False))
        
        # Repeat z for each candidate B_i: shape becomes [B, C, latent_dim]
        z_repeated = z.unsqueeze(1).expand(b, c, z.shape[-1])
        B_flat_repeated = B_flat.unsqueeze(0).expand(b, c, 9)
        #B_flat_norm = self.norm2(B_flat_repeated)
        zB = torch.cat([z_repeated, 0.3 * B_flat_repeated], dim=2)
        zB = self.norm(zB)
        # The rotation head expects input shape [B, C, input_dim] and outputs R of shape [B, C, 3, 3]
        R = self.rotation_head(zB)
        
        # Calculate mean and std deviation of R across B, where C is 1
        R_mean = R.mean(dim=0)
        R_std = R.std(dim=0)
        # print("Mean of R across B:", R_mean)
        # print("Std deviation of R across B:", R_std)
        return R, B_candidates, z

            



