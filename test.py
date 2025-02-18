import torch
import torch

def explicit_theta_to_B(theta, angles_in_degrees=False):
    """
    Compute the reciprocal lattice matrix B(θ) explicitly without matrix inversion.

    Args:
        theta (torch.Tensor): Tensor of shape (6,) containing:
                              (a, b, c, alpha, beta, gamma)
        angles_in_degrees (bool): If True, convert input angles from degrees to radians.

    Returns:
        B (torch.Tensor): Reciprocal lattice matrix (3x3).
    """
    a, b, c, alpha, beta, gamma = theta  # Unpack lattice parameters

    # Convert angles to radians if needed
    if angles_in_degrees:
        alpha = alpha * torch.pi / 180.0
        beta  = beta  * torch.pi / 180.0
        gamma = gamma * torch.pi / 180.0

    # Compute cosines and sines
    cos_alpha, cos_beta, cos_gamma = torch.cos(alpha), torch.cos(beta), torch.cos(gamma)
    sin_alpha, sin_beta, sin_gamma = torch.sin(alpha), torch.sin(beta), torch.sin(gamma)

    # Compute unit cell volume
    V = a * b * c * torch.sqrt(1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2 + 
                                2 * cos_alpha * cos_beta * cos_gamma)

    # Ensure no division by zero
    eps = 1e-8
    sin_gamma = torch.clamp(sin_gamma, min=eps)

    # Construct reciprocal basis matrix B explicitly
    B = torch.zeros((3, 3), dtype=theta.dtype, device=theta.device)

    B[0, 0] = (b * c * sin_alpha) / V
    B[0, 1] = (c * (cos_alpha * cos_gamma - cos_beta)) / (V * sin_gamma)
    B[0, 2] = (a * (cos_beta * cos_gamma - cos_alpha)) / (V * sin_gamma)

    B[1, 1] = (a * c * sin_beta) / (V * sin_gamma)
    B[1, 2] = (a * (cos_beta * cos_gamma - cos_alpha)) / (V * sin_gamma)

    B[2, 2] = (a * b * sin_gamma) / V

    return B

import torch

def compute_B_matrix(theta):
    """
    Compute the B matrix for a given unit cell.
    
    Args:
        a (float): Unit cell parameter a (Å)
        b (float): Unit cell parameter b (Å)
        c (float): Unit cell parameter c (Å)
        alpha (float): Angle α in radians
        beta (float): Angle β in radians
        gamma (float): Angle γ in radians
    
    Returns:
        torch.Tensor: 3x3 B matrix
    """
    a, b, c, alpha, beta, gamma = theta
    cos_alpha, cos_beta, cos_gamma = torch.cos(alpha), torch.cos(beta), torch.cos(gamma)
    sin_alpha, sin_beta, sin_gamma = torch.sin(alpha), torch.sin(beta), torch.sin(gamma)

    # Compute unit cell volume V
    V = a * b * c * torch.sqrt(1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2 + 2 * cos_alpha * cos_beta * cos_gamma)

    # Compute reciprocal lattice parameters
    a_star = (b * c * sin_alpha) / V
    b_star = (a * c * sin_beta) / V
    c_star = (a * b * sin_gamma) / V

    # Compute reciprocal angles
    cos_alpha_star = (cos_beta * cos_gamma - cos_alpha) / (sin_beta * sin_gamma)
    cos_beta_star = (cos_alpha * cos_gamma - cos_beta) / (sin_alpha * sin_gamma)
    cos_gamma_star = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta)
    sin_alpha_star = torch.sqrt(1 - cos_alpha_star**2)
    sin_beta_star = torch.sqrt(1 - cos_beta_star**2)
    sin_gamma_star = torch.sqrt(1 - cos_gamma_star**2)

    # Construct B matrix
    B = torch.tensor([
        [a_star, b_star * cos_gamma_star, c_star * cos_beta_star],
        [0, b_star * sin_gamma_star, -c_star * sin_beta_star],
        [0, 0, c_star * sin_alpha_star]
    ], dtype=torch.float32)

    return B


def theta_to_B(theta): 
    """
    Convert unit cell parameters θ into the reciprocal
    lattice basis matrix B(θ) for a triclinic unit cell.
    
    Args:
        theta: 6x1 tensor of unit cell parameters (a, b, c, α, β, γ).
    
    Returns:
        B (torch.Tensor): Reciprocal basis matrix of shape (3, 3) such that
                        q = R @ B @ h.
    """
    a, b, c, alpha, beta, gamma = theta 

    cos_alpha = torch.cos(alpha)
    cos_beta  = torch.cos(beta)
    cos_gamma = torch.cos(gamma)
    
    eps = 1e-8
    sin_gamma = torch.clamp(torch.sin(gamma), min=eps)
    
    A = torch.zeros((3, 3), dtype=theta.dtype)
    A[0, 0] = a
    A[0, 1] = b * cos_gamma
    A[0, 2] = c * cos_beta
    
    A[1, 1] = b * sin_gamma
    A[1, 2] = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    
    V = a * b * c * torch.sqrt(1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2 + 2 * cos_alpha * cos_beta * cos_gamma)
    A[2, 2] = V / (a * b * sin_gamma)
    
    A_inv = torch.linalg.inv(A)
    B = A_inv.T
    
    return B

import torch

def compute_O_matrix(theta):
    """
    Compute the transformation matrix O given unit cell parameters.
    
    Parameters:
    theta: torch.Tensor of shape (6,), containing [a, b, c, alpha, beta, gamma] in radians.
    
    Returns:
    O: torch.Tensor of shape (3,3), the transformation matrix.
    """
    a, b, c, alpha, beta, gamma = theta
    
    cos_alpha, cos_beta, cos_gamma = torch.cos(alpha), torch.cos(beta), torch.cos(gamma)
    sin_gamma = torch.sin(gamma)
    
    V = a * b * c * torch.sqrt(1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2 + 2 * cos_alpha * cos_beta * cos_gamma)
    
    A = torch.tensor([
        [a, b * cos_gamma, c * cos_beta],
        [0, b * sin_gamma, c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma],
        [0, 0, V / (a * b * sin_gamma)]
    ])
    A_inv = torch.linalg.inv(A)
    B = A_inv.T
    
    return B
    
def compute_O_inverse(theta):
    """
    Compute the inverse of the transformation matrix O given unit cell parameters.
    
    Parameters:
    theta: torch.Tensor of shape (6,), containing [a, b, c, alpha, beta, gamma] in radians.
    
    Returns:
    O_inv: torch.Tensor of shape (3,3), the inverse transformation matrix.
    """
    a, b, c, alpha, beta, gamma = theta
    
    cos_alpha, cos_beta, cos_gamma = torch.cos(alpha), torch.cos(beta), torch.cos(gamma)
    sin_gamma = torch.sin(gamma)
    
    V = a * b * c * torch.sqrt(1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2 + 2 * cos_alpha * cos_beta * cos_gamma)
    
    a, b, c, alpha, beta, gamma = theta 

    cos_alpha = torch.cos(alpha)
    cos_beta  = torch.cos(beta)
    cos_gamma = torch.cos(gamma)
    sin_gamma = torch.clip(torch.sin(gamma), min=1e-2)
    V = a * b * c * torch.sqrt(1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2 + 2 * cos_alpha * cos_beta * cos_gamma)
    
    A_inv = torch.zeros((3, 3), dtype=theta.dtype)
    A_inv[0, 0] = 1/a
    A_inv[0, 1] = -cos_gamma / (a * sin_gamma)
    A_inv[0, 2] = ((b * cos_gamma * c * (cos_alpha - cos_beta * cos_gamma)) / sin_gamma - b * c * cos_beta * sin_gamma) / V
    
    A_inv[1, 1] = 1 / (b * sin_gamma)
    A_inv[1, 2] = (-a * c * (cos_alpha - cos_beta * cos_gamma)) / (V * sin_gamma)
    
    A_inv[2, 2] = (a * b * sin_gamma) / V
    
    B = A_inv.T
    
    return B




# Example usage
theta=torch.tensor([1.0, 2.0, 3.0, torch.pi/3, torch.pi/3, torch.pi/3])
theta=torch.tensor([1,1,1,1,1,1])
# print(theta_to_B(theta))
print()
print(compute_O_matrix(theta))
print()
print(compute_O_inverse(theta))