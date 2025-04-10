import torch
import torch.nn as nn

class EncoderDS(nn.Module):
    """
    The encoder model takes in the reciprocal space vectors q_i (3x1) of a diffraction image Q_i and
    outputs a latent representation z (d x 1) of the diffraction image.
    NOTE: other options include PointNet, PointNet++, and set transformers.
    Args:
        phi (nn.Sequential): Feature extractor applied to each individual q_i.
        rho (nn.Sequential): Aggregator applied after pooling over all q_i.
    """
    
    def __init__(self, input_dim=3, hidden_dim=128, output_dim=64):
        """
        Args:
            input_dim: Dimensionality of each reciprocal vector.
            hidden_dim: Number of hidden units in phi network.
            output_dim: Dimensionality of the latent vector z.
        """
        super().__init__()        
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.PReLU(),  
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU()
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
            # nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )
        # self.dropout = nn.Dropout(p=0.3)

        
    def forward(self, Q_i, mask=None):
        """
        Args:
            Q_i (Nx3): Tensor of shape (N, 3) where N is the number of q vectors.
        Returns:
            z (torch.Tensor): Latent representation of Q_i (fixed size: output_dim).
        """
        phi_Q_i = self.phi(Q_i) #BxNx3 -> BxNx128
        #print("Mean and std of phi_Q_i:", phi_Q_i.mean(dim=0), phi_Q_i.std(dim=0))
        if mask is None:
            mask = torch.ones(Q_i.shape[0], Q_i.shape[1], device=Q_i.device, dtype=Q_i.dtype)
            
        phi_Q_i = phi_Q_i * mask.unsqueeze(-1) #Apply mask
        # phi_Q_i = self.dropout(phi_Q_i)
        pooled = torch.sum(phi_Q_i, dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        z = self.rho(pooled) 
        #print("Mean and std of z:", z.mean(dim=0), z.std(dim=0))

        return z
    
if __name__ == '__main__':
    # Test Encoder
    encoder = EncoderDS()
    Q_i = torch.randn(3,10, 3) #BxNx3
    z = encoder(Q_i)
    print(z)
    print(z.shape)
        

