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
from loss_functions.liealgebra_loss import LieAlgebraLoss
from loss_functions.symmetry_loss import SymmetryAwareLossLoop
from loss_functions.frobenius_loss import FrobeniusLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import gemmi


def rotation_angle_error(
    R_pred: torch.Tensor,   # shape (B, C, 3, 3)
    R_gt: torch.Tensor,     # shape (B, 3, 3)
    symm_group="P 61 2 2",
    device="cpu",
    eps=1e-6
) -> torch.Tensor:
    B, C, _, _ = R_pred.shape
    sg = gemmi.SpaceGroup(symm_group)
    rot_mats = torch.stack([
        torch.tensor(np.array(op.rot, dtype=float) / op.DEN, dtype=torch.float32, device=device)
        for op in sg.operations()
    ])
    S = rot_mats.shape[0]
    R_gt_exp = R_gt[:, None, None, :, :]
    R_pred_exp = R_pred[:, :, None, :, :]
    symm_exp = rot_mats[None, None, :, :, :]
    R_gt_sym = torch.matmul(symm_exp, R_gt_exp)
    R_rel = torch.matmul(R_pred_exp, R_gt_sym.transpose(-2, -1))
    trace = R_rel.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos_theta)
    min_over_sym = theta.min(dim=2)[0]
    min_over_cand = min_over_sym.min(dim=1)[0]
    return min_over_cand.mean()



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
        
        # Normalize globally
        Q_i = (Q_i - self.global_mean) / (self.global_std + self.eps)
        
        # Normalize per-sample
        # sample_mean = np.mean(Q_i, axis=0)
        # sample_std = np.std(Q_i, axis=0) + 1e-6  
        # Q_i = (Q_i - sample_mean) / sample_std
        
        Q_i = torch.tensor(Q_i, dtype=torch.float32)
        U_i = torch.tensor(U_i, dtype=torch.float32)
        return Q_i, U_i


def supervised_collate(batch, device=torch.device("cpu")):
    Q_tuple, U_tuple = zip(*batch)
    padded_Q, lengths, mask = collate_fn(list(Q_tuple), device=device)
    U_tensor = torch.stack(U_tuple).to(device)
    return padded_Q, lengths, mask, U_tensor


def load_train_val_data(dataset_path, batch_size=4, val_ratio=0.2, device=torch.device("cpu")):
    full_dataset = Supervised_QtoR_DiffractionDataset(dataset_path)
    n = len(full_dataset)
    val_size = int(n * val_ratio)
    train_size = n - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    collate = lambda batch: supervised_collate(batch, device=device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate, shuffle=False)
    return train_loader, val_loader



def train_QtoR_supervised(model, dataset_path, cell, num_epochs=1, batch_size=3, lr=1e-3, weight_decay=1e-4, device='cpu'):
    train_loader, val_loader = load_train_val_data(dataset_path, batch_size=batch_size, device=device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.to(device)
    model.train()
    #loss_function = GeodesicLoss()
    #loss_function = LieAlgebraLoss()
    loss_function = SymmetryAwareLossLoop(base_loss=GeodesicLoss(reduction='none'), cell=cell)
    best_val_loss = float('inf')
    best_train_loss = 0.95
    log_file = "training_log.txt"
    #scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        errors_train, errors_val = [], []
        model.train()
        for padded_Q, lengths, mask, U_gt in train_loader:
            padded_Q = padded_Q
            mask = mask
            U_gt = U_gt
            
            optimizer.zero_grad()
            R_candidates, _, _ = model(padded_Q, mask)
            R_pred = R_candidates[:, :, :, :] 

            loss = loss_function(R_pred, U_gt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            errors_train.append(loss.item())
            
        avg_train_loss = epoch_loss / len(train_loader)

        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for padded_Q, lengths, mask, U_gt in val_loader:
                padded_Q = padded_Q
                mask = mask
                U_gt = U_gt   
                R_candidates, _, _ = model(padded_Q, mask)
                R_pred = R_candidates[:,:,:,:] # if geodesic or lie use 0 for axis 1
                loss_val = loss_function(R_pred, U_gt)
                val_loss += loss_val.item()
                errors_val.append(loss_val.item())
 
        avg_val_loss = val_loss / len(val_loader)
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # print(f"mean and std of train loss: {np.mean(errors_train):.4f}, {np.std(errors_train):.4f}")
            # print(f"max and min of train loss: {np.max(errors_train):.4f}, {np.min(errors_train):.4f}")
            # print(f"mean and std of val loss: {np.mean(errors_val):.4f}, {np.std(errors_val):.4f}")
            # print(f"max and min of val loss: {np.max(errors_val):.4f}, {np.min(errors_val):.4f}")
            
            # print(errors_train)
            # print("#"*20)
            # print(errors_val)
            # print("]"*20)
            
            with open(log_file, "a") as f:
                log_message = f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
                f.write(log_message + "\n") 
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            torch.save(model.state_dict(), "models_params_saved/qtor_model_best.pth")

        #scheduler.step()
    print("Finished training.")

def train():
    dataset_path = "data/output_data.npy"  

    unit_cells = np.array([entry["Unit_Cell"] for entry in np.load("data/output_data.npy", allow_pickle=True)])
    m = torch.tensor(np.mean(unit_cells, axis=0), dtype=torch.float32)
    s = torch.tensor(np.std(unit_cells, axis=0), dtype=torch.float32)
    print(m,s)
    theta_as_param = False

    model = QtoRModel(latent_dim=128, num_theta_samples=2, encoder_hidden=128, rotation_hidden=128,
                        theta_isParam=theta_as_param, theta_mu=m, theta_diagS=s, use_fourier=True, fourier_mapping_size=16, fourier_scale=10.0)
    #model.load_state_dict(torch.load("qtor_model_best.pth"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_QtoR_supervised(model, dataset_path, cell=m, num_epochs=6000, batch_size=64, lr=1e-4, device=device)  

if __name__ == "__main__":
    train()
    