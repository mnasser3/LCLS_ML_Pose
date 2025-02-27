import torch
import torch.nn as nn
import torch.nn.functional as F

class AssignHead(nn.Module):
    """
    Predicts Miller indices (h) for each reflection using a Gumbel Softmax approach.
    Uses LazyLinear layers to allow flexible input dimensions (e.g., sometimes including
    extra features such as B and R, sometimes not).
    
    The output is a soft one-hot distribution for each of the 3 Miller index coordinates,
    which is then converted to an expected value over the discrete set [-10, 10].
    
    Args:
        hidden_dim (int): Number of hidden units in the intermediate layer.
        num_classes (int): Number of classes per Miller index coordinate (default: 21, for [-10,10]).
        temperature (float): Temperature parameter for Gumbel Softmax (default: 1.0).
    """
    def __init__(self, hidden_dim=128, N_max=10, temperature=1.0):
        super().__init__()
        self.num_classes = 2*N_max + 1
        self.temperature = temperature
        self.fc1 = nn.LazyLinear(hidden_dim)
        self.fc2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, 3 * self.num_classes)
        )
        self.register_buffer('index_values', torch.linspace(-10, 10, steps=self.num_classes))
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features of shape (B, N, input_dim), where B is the batch size and N is the number of reflections.
            NOTE: input_dim is not fixed as we may add B(theta) and R.
        
        Returns:
            h (torch.Tensor): Predicted Miller indices (expected values), shape (B, N, 3).
            soft_indices (torch.Tensor): Soft one-hot distributions, shape (B, N, 3, num_classes).
        """
        b,c, N, _ = x.shape
        x = self.fc1(x) #(B,C, N, hidden_dim)
        logits = self.fc2(x) #(B,C, N, 3*num_classes)
        logits = logits.view(b,c, N, 3, self.num_classes)  # Reshape to (B,C, N, 3, num_classes)
        
        soft_indices = F.gumbel_softmax(logits, tau=self.temperature, hard=False, dim=-1)
        h = (soft_indices @ self.index_values.unsqueeze(-1)).squeeze(-1) # (B,C,N,3)
  
        return h

# Example usage:
if __name__ == "__main__":
    batch_size = 2
    num_reflections = 6
    input_features = torch.randn(batch_size, num_reflections, 3)
    
    assign_head = AssignHead(hidden_dim=128, N=10, temperature=0.15)
    h, soft_indices = assign_head(input_features)
    print("Predicted Miller indices:", h)  
    print("Predicted Miller indices shape:", h.shape)       
