import torch
from torch.utils.data import Dataset
import numpy as np

class DiffractionDataset(Dataset):
    def __init__(self, dataset_path):
        """
        Initialize the dataset (Mx(Nx3)) with the path to the dataset. M is the number of samples,
        and N is the number of reciprocal space vectors per diffraction image, and 3 is the dimension of each vector.
        NOTE: N is not fixed, it can be different for each diffraction image.
        
        Args:
            dataset_path (_str_): path to the dataset
        """
        self.data = np.load(dataset_path)
        assert isinstance(self.data, (list, np.ndarray))
        for Q in self.data: # Q is a diffraction image of size Nx3
            assert Q.shape[1] == 3 and Q.ndim == 2
        
    def __len__(self):
        """
        returns the number of samples in the dataset
        """
        return len(self.data)

    def __getitem__(self, i):
        """
        returns the i_th diffraction image (Nx3) in the dataset
        """
        Q_i = self.data[i]
        Q_i = torch.tensor(Q_i, dtype=torch.float32)
        return Q_i
    