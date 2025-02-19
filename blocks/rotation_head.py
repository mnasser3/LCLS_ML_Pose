import torch
import torch.nn as nn
from blocks.utils import *

class RotationHead(nn.Module):
    """
    The rotation head model takes in the latent representation z along with flattened B_i and
    outputs the rotation matrix SO(3) (using 6D rotation representation) that predicts the orientation of the unit cell.
    The process follows:
        1. Map the input latent vector to a 6D vector.
        2. Split the 6D vector into two 3D vectors a1 and a2.
        3. Normalize a1 to get b1.
        4. Orthogonalize a2 with respect to b1 and normalize to get b2.
        5. Compute b3 as the cross product of b1 and b2.
        6. Form the rotation matrix R = [b1, b2, b3].

    Args:
        d (int): Dimensionality of the latent vector z.
    """
    
    def __init__(self, input_dim=137,hidden_dim=128):
        """
        Args:
            input_dim (int): Dimensionality of z.
            hidden_dim (int): Number of hidden units in the intermediate layer.
        """
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 6)  # Output 6D vector
        )
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input latent vectors of shape (B,C, input_dim), 
                            where B is the batch size and C number of samples of theta.
        
        Returns:
            R (torch.Tensor): Rotation matrix of shape (B,C, 3, 3).
        """
        sixd = self.fc(x)  # Shape: (B,C, 6)
        print(sixd.shape)
        B, C, _ = sixd.shape
        # Flatten the candidate dimension into the batch dimension
        ortho6d_flat = sixd.view(B * C, 6)
        # Compute rotation matrices for all candidates
        R_flat = compute_rotation_matrix_from_ortho6d(ortho6d_flat)
        # Reshape back to [B, C, 3, 3]
        R = R_flat.view(B, C, 3, 3)
        return R
    
    
    """ Helper functions """
        
    def sixd_to_R(self,ortho6d):
        """
        Convert a batch of 6D rotation representations into 3x3 rotation matrices.
        
        Args:
            ortho6d (torch.Tensor): Tensor of shape [batch, 6].
            
        Returns:
            torch.Tensor: Rotation matrices of shape [batch, 3, 3].
        """
        # Split the 6D vector into two 3D vectors
        x_raw = ortho6d[:, 0:3]
        y_raw = ortho6d[:, 3:6]
        
        # Normalize the first vector
        x = normalize_vector(x_raw)
        
        # Make y_raw orthogonal to x via cross product, then normalize to get z
        z = cross_product(x, y_raw)
        z = normalize_vector(z)
        
        # Get y by taking the cross product of z and x
        y = cross_product(z, x)
        
        # Reshape each vector to [batch, 3, 1] and concatenate to form a rotation matrix
        x = x.unsqueeze(2)  # [batch, 3, 1]
        y = y.unsqueeze(2)  # [batch, 3, 1]
        z = z.unsqueeze(2)  # [batch, 3, 1]
        
        rotation_matrix = torch.cat((x, y, z), dim=2)  # [batch, 3, 3]
        return rotation_matrix

        
if __name__ == "__main__":
    # Suppose we have a latent representation of shape (batch_size, input_dim)
    big_batch_size = 1
    batch_size = 3
    input_dim = 64
    latent = torch.randn(big_batch_size,batch_size, input_dim)
    
    # Instantiate the RotationHead
    rot_head = RotationHead(input_dim=input_dim, hidden_dim=128)
    
    # Get the rotation matrices
    R = rot_head(latent)
    # print("Rotation Matrices Shape:\n", R)
    

    def is_SO3(R, atol=1e-6):
        """Check if R is in SO(3): R^T R = I and det(R) = 1"""
        I = torch.eye(3, device=R.device, dtype=R.dtype)
        orthogonality = torch.allclose(R @ R.T, I, atol=atol)
        determinant = torch.allclose(torch.det(R), torch.tensor(1.0, device=R.device, dtype=R.dtype), atol=atol)
        return orthogonality and determinant

    for i in R[0]:
        print(i)
        print(is_SO3(i))
        print()




