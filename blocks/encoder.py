import torch
import torch.nn as nn

class Encoder(nn.Module):
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
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        
        self.rho = nn.Sequential(          
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
    def forward(self, Q_i, mask):
        """
        Args:
            Q_i (Nx3): Tensor of shape (N, 3) where N is the number of q vectors.
        Returns:
            z (torch.Tensor): Latent representation of Q_i (fixed size: output_dim).
        """
        phi_Q_i = self.phi(Q_i) #BxNx3 -> BxNx128
        phi_Q_i = phi_Q_i * mask.unsqueeze(-1) #Apply mask
        pooled = torch.sum(phi_Q_i, dim=1)  
        pooled = pooled / mask.sum(dim=1, keepdim=True).clamp(min=1) #Normalized average pooling
        z = self.rho(pooled) 
        return z
    
if __name__ == '__main__':
    # Test Encoder
    encoder = Encoder()
    Q_i = torch.randn(3,10, 3) #BxNx3
    z = encoder(Q_i)
    print(z)
    print(z.shape)
        

