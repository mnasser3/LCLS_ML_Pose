import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from blocks.diffraction_dataset import DiffractionDataset
from QR.utils import collate_fn
import numpy as np
from tqdm import tqdm
from QR.QRmodel import QtoRModel
from loss_functions.geodesic_loss import GeodesicLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

class Supervised_QtoR_DiffractionDataset(DiffractionDataset):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        self.full_data = np.load(dataset_path, allow_pickle=True)
        all_Q = []
        for sample in self.full_data:
            Q_i = np.array(sample["Q"], dtype=np.float32)
            all_Q.append(Q_i)
        all_Q = np.concatenate(all_Q, axis=0) 
        self.global_mean = np.mean(all_Q, axis=0)
        self.global_std = np.std(all_Q, axis=0)
        self.eps = 1e-6
        
    def __getitem__(self, i):
        sample = self.full_data[i]
        Q_i = sample["Q"]
        U_i = sample["U"] 
        Q_i = np.array(Q_i, dtype=np.float32)
        U_i = np.array(U_i, dtype=np.float32)
        Q_i = (Q_i - self.global_mean) / (self.global_std + self.eps)
        Q_i = torch.tensor(Q_i, dtype=torch.float32)
        U_i = torch.tensor(U_i, dtype=torch.float32)
        return Q_i, U_i


def supervised_collate(batch):
    Q_tuple, U_tuple = zip(*batch)
    padded_Q, lengths, mask = collate_fn(list(Q_tuple))
    U_tensor = torch.stack(U_tuple)
    return padded_Q, lengths, mask, U_tensor

def load_train_val_data(dataset_path, batch_size=4, val_ratio=0.2):
    full_dataset = Supervised_QtoR_DiffractionDataset(dataset_path)
    n = len(full_dataset)
    val_size = int(n* val_ratio)
    train_size = n - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=supervised_collate, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=supervised_collate, shuffle=False)
    return train_loader, val_loader


def train_QtoR_supervised(model, dataset_path, num_epochs=1, batch_size=3, lr=1e-3, device='cpu'):
    train_loader, val_loader = load_train_val_data(dataset_path, batch_size=batch_size, val_ratio=0.2)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.to(device)
    model.train()
    loss_function = GeodesicLoss()
    best_val_loss = float('inf')
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        model.train()
        for padded_Q, lengths, mask, U_gt in train_loader:
            padded_Q = padded_Q.to(device)  
            mask = mask.to(device)
            U_gt = U_gt.to(device)      
            
            optimizer.zero_grad()
            R_candidates, _, _ = model(padded_Q, mask)
            R_pred = R_candidates[:,0,:,:]  
            

            # print("R_pred:", R_pred[0])
            # print("U_gt:", U_gt[0])

                    
            loss = loss_function(R_pred, U_gt)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # print("Loss:", loss.item())
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f"{name} gradient norm: {param.grad.norm()}")
            
        avg_train_loss = epoch_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for padded_Q, lengths, mask, U_gt in val_loader:
                padded_Q = padded_Q.to(device)
                mask = mask.to(device)
                U_gt = U_gt.to(device)
                R_candidates, _, _ = model(padded_Q, mask)
                R_pred = R_candidates[:,0,:,:] 
                loss_val = loss_function(R_pred, U_gt)
                val_loss += loss_val.item()
        avg_val_loss = val_loss / len(val_loader)
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "qtor_model_best.pth")

        scheduler.step()
    print("Finished training.")
    
if __name__ == "__main__":
    def train():
        dataset_path = "data/output_data.npy"  
        train_data,val_data = load_train_val_data(dataset_path, batch_size=16, val_ratio=0.2)

        unit_cells = np.array([entry["Unit_Cell"] for entry in np.load("data/output_data.npy", allow_pickle=True)])

        m = torch.tensor(np.mean(unit_cells, axis=0), dtype=torch.float32)
        s = torch.tensor(np.std(unit_cells, axis=0), dtype=torch.float32)
        print(m,s)
        theta_as_param = False

        model = QtoRModel(latent_dim=128, num_theta_samples=2, encoder_hidden=128, rotation_hidden=128,
                            theta_isParam=theta_as_param, theta_mu=m, theta_diagS=s)

        train_QtoR_supervised(model, dataset_path, num_epochs=1000, batch_size=32, lr=1e-2, device='cpu')

        print(f"Number of training samples: {len(train_data.dataset)}")
        print(f"Number of validation samples: {len(val_data.dataset)}")

    train()
    
#2.11,2.14

    # # Example usage of training with trivial data:

    # import torch
    # from torch.utils.data import Dataset

    # class TrivialQtoRDataset(Dataset):
    #     def __init__(self, num_samples=1000, Q_shape=(8, 3)):  # Use (8, 3) instead of (3,)
    #         self.num_samples = num_samples
    #         self.Q_shape = Q_shape

    #     def __len__(self):
    #         return self.num_samples

    #     def __getitem__(self, idx):
    #         Q = torch.zeros(self.Q_shape)  # or any trivial constant values
    #         U = torch.eye(3, dtype=torch.float32)
    #         return Q, U


    # def load_trivial_data(batch_size=8, num_samples=1000):
    #     dataset = TrivialQtoRDataset(num_samples=num_samples, Q_shape=(8,3))
    #     loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    #     return loader

    # # Example usage in training:
    # def train_trivial(model, num_epochs=10, batch_size=8, lr=1e-3, device='cpu'):
    #     train_loader = load_trivial_data(batch_size=batch_size, num_samples=1000)
    #     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #     model.to(device)
    #     model.train()
    #     loss_func  = GeodesicLoss()
    #     for epoch in range(num_epochs):
    #         epoch_loss = 0.0
    #         for Q, U_gt in train_loader:
    #             Q = Q.to(device)
    #             U_gt = U_gt.to(device)
                
    #             optimizer.zero_grad()
    #             # Make sure your model outputs a rotation matrix.
    #             R_candidates, _, _ = model(Q)  # Adjust if your model expects additional inputs.
    #             R_pred = R_candidates[:,0,:,:]

    #             loss = loss_func(R_pred, U_gt)
    #             loss.backward()
    #             optimizer.step()
    #             epoch_loss += loss.item()
    #         avg_loss = epoch_loss / len(train_loader)
    #         print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
            
    #     print("Finished training on trivial data.")

    # unit_cells = np.array([entry["Unit_Cell"] for entry in np.load("data/output_data.npy", allow_pickle=True)])

    # m = torch.tensor(np.mean(unit_cells, axis=0), dtype=torch.float32)
    # s = torch.tensor(np.std(unit_cells, axis=0), dtype=torch.float32)
    # theta_as_param = False
    # model = QtoRModel(latent_dim=128, num_theta_samples=2, encoder_hidden=128, rotation_hidden=128,
    #                     theta_isParam=theta_as_param, theta_mu=m, theta_diagS=s)
    # train_trivial(model, num_epochs=30, batch_size=8, lr=1e-3)