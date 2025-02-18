import torch

def normalize_vector(v, return_mag=False, eps=1e-8):
    """
    Normalize a batch of vectors.
    
    Args:
        v (torch.Tensor): Tensor of shape [batch, n].
        return_mag (bool): If True, also return the computed magnitudes.
        eps (float): Small constant for numerical stability.
        
    Returns:
        torch.Tensor: Normalized vectors of shape [batch, n].
        (Optional) torch.Tensor: Magnitudes of the input vectors.
    """
    v_mag = torch.norm(v, dim=1, keepdim=True).clamp(min=eps)
    v_norm = v / v_mag
    if return_mag:
        return v_norm, v_mag.squeeze(1)
    return v_norm

def cross_product(u, v):
    """
    Compute the cross product between two batches of 3D vectors.
    
    Args:
        u (torch.Tensor): Tensor of shape [batch, 3].
        v (torch.Tensor): Tensor of shape [batch, 3].
    
    Returns:
        torch.Tensor: Cross product, shape [batch, 3].
    """
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    return torch.stack((i, j, k), dim=1)

def compute_rotation_matrix_from_ortho6d(ortho6d):
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
