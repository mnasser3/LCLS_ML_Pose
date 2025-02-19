from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import numpy as np
import torch

def collate_fn(batch):
    """
    Collate function that pads variable-length Q_i tensors.
    Each Q_i is of shape (N_i, 3).

    Returns:
        padded_Q: Tensor of shape (B, max_length, 3)
        lengths: Tensor of original lengths for each Q_i.
        mask: Tensor of shape (B, max_length) where True indicates a valid vector and False is padding.
    """
    device = batch[0].device
    lengths = torch.tensor([q.shape[0] for q in batch], dtype=torch.long, device=device)
    padded_Q = pad_sequence(batch, batch_first=True, padding_value=0.0).to(device)
    max_length = padded_Q.size(1)
    mask = torch.arange(max_length, device=device)[None, :] < lengths[:, None]
    return padded_Q, lengths, mask

def data_gen(B,N):
    data = []
    for i in range(B):
        N = np.random.randint(3, 5)
        diffraction_image = np.random.randn(N, 3)
        data.append(diffraction_image)
    np.save("QR/test_diffraction_data.npy", np.array(data, dtype=object))