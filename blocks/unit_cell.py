import torch
import torch.nn as nn
import torch.distributions as dist

class UnitCell(nn.Module):
    """
    Variational Inference (VI) model for unit cell parameters θ.
    
    θ = (a, b, c, α,β,γ) is modeled as a variable with learnable mean and standard deviation
    #NOTE: alpha, beta, gamma are in radians
    Attributes:
        mu_theta (torch.nn.Parameter): Mean of the variational distribution.
        log_sigma_theta (torch.nn.Parameter): Log of standard deviation (to ensure positivity).
    """
    def __init__(self,mu=None,diag_S=None,isParam=True,num_samples=3):
        """
        Args:
            mu (6x1): prior mean vector. Defaults to None.
            diag_S (6x1): diagonal covariance, assuming cov(X,Y)=0. Defaults to None.
            given (bool): determines if unit cell parameters are given or not. Defaults to False.
        """
        super().__init__()
        self.isParam=isParam
        self.num_samples=num_samples
        
        if self.isParam: #if unit cell parameters are to be learned
            if mu is not None and diag_S is not None: # if prior is given
                assert diag_S.dim()==1
                logS = torch.log(diag_S)
            else: # if prior is not given
                #TODO: LOG-NORMAL DISTRIBUTION
                mu = torch.tensor([86.22, 95.07, 117.53, 89.985, 93.626, 95.41], dtype=torch.float32)
                diag_S = torch.tensor([ 48.275, 49.23, 75.38, 2.81, 8.2, 11.98], dtype=torch.float32)
                logS = torch.log(diag_S)
            self.p_dist= dist.Normal(mu, torch.sqrt(diag_S))
            self.q_mu = nn.Parameter(mu)
            self.q_logS = nn.Parameter(logS)
                
        else: # if unit cell parameters are given
            assert mu is not None
            self.register_buffer('fixed_theta', torch.as_tensor(mu, dtype=torch.float32)) # register_buffer to disable backpropagation on theta
            
    
    def forward(self):
        if self.isParam is False: # if unit cell parameters are given
            B = self.theta_to_B(self.fixed_theta.unsqueeze(0))
            #O = self.theta_to_O(self.fixed_theta.unsqueeze(0))
            return B, None, None
        else: #TODO: Log-Normal Distribution
            self.q_dist = dist.Normal(self.q_mu, torch.sqrt(torch.exp(self.q_logS)))
            thetas_sampled = self.q_dist.rsample((self.num_samples,))
            B = self.theta_to_B(thetas_sampled)
            #O = self.theta_to_O(thetas_sampled)
            return B, self.q_mu, self.q_logS
    
    def theta_to_B(self,theta): 
        """
        Convert unit cell parameters θ into the reciprocal
        lattice basis matrix B(θ) for a triclinic unit cell.
        
        Args:
            theta: 6x1 tensor of unit cell parameters (a, b, c, α, β, γ).
        
        Returns:
            B (torch.Tensor): Reciprocal basis matrix of shape (3, 3) such that
                            q = R @ B @ h.
            B = A_inv.T
            A = [a, bcosγ, c*cosβ],
                [0, bsinγ, c(cosα - cosβcosγ)/sinγ],
                [0, 0, V/(a*b*sinγ)]
        """
        a, b, c, alpha, beta, gamma = theta[:, 0], theta[:, 1], theta[:, 2], theta[:, 3], theta[:, 4], theta[:, 5]

        cos_alpha = torch.cos(alpha)
        cos_beta  = torch.cos(beta)
        cos_gamma = torch.cos(gamma)
        sin_gamma = torch.sin(gamma)
        
        eps=1e-3
        a,b,c = torch.clamp(a,min=eps),torch.clamp(b,min=eps),torch.clamp(c,min=eps)
        vol_sqrt_term = 1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2 + 2 * cos_alpha * cos_beta * cos_gamma
        vol_sqrt_term= torch.clamp(vol_sqrt_term, min=eps)
        V = a * b * c * torch.sqrt(vol_sqrt_term)
        V = torch.clamp(V, min=eps)
        sin_gamma = torch.sign(sin_gamma)*torch.clamp(torch.abs(sin_gamma), min=eps) 
        
        A_inv = torch.zeros((theta.shape[0],3, 3), dtype=theta.dtype, device=theta.device)
        A_inv[:,0, 0] = 1/a
        A_inv[:,0, 1] = -cos_gamma / (a * sin_gamma)
        A_inv[:,0, 2] = ((b * cos_gamma * c * (cos_alpha - cos_beta * cos_gamma)) / sin_gamma - b * c * cos_beta * sin_gamma) / V
        
        A_inv[:,1, 1] = 1 / (b * sin_gamma)
        A_inv[:,1, 2] = (-a * c * (cos_alpha - cos_beta * cos_gamma)) / (V * sin_gamma)
        
        A_inv[:,2, 2] = (a * b * sin_gamma) / V
        if A_inv.shape[0]==1:
            A_inv=A_inv.squeeze(0)
            B = A_inv.T
            B=B.unsqueeze(0)
        else:
            B = A_inv.mT
        return B
    

    def theta_to_O(self, theta):
        """
        Convert unit cell parameters θ into the real-space basis matrix O(θ)
        for a triclinic unit cell.

        O is given by the 3×3 matrix:
            [ a,            b*cosγ,                                    c*cosβ              ]
            [ 0,            b*sinγ,    c*(cosα - cosβ*cosγ) / sinγ                           ]
            [ 0,            0,         V / (a*b*sinγ)                                         ]

        with
  