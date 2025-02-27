from QH.QHmodel import *
from QH.utils import *
from QR.QRmodel import *
    
if __name__ == '__main__':
    B = 3
    C=2
    N = 4
    data_gen(B,3,7,"QH")  # B,N,3 (N varying per batch)
    dataset = DiffractionDataset("QH/test_diffraction_data.npy")
    dataloader = DataLoader(dataset, batch_size=B, collate_fn=collate_fn)

    # for padded_Q, lengths, mask in dataloader:
    #     print(f"Padded Q shape: {padded_Q.shape}")  
    #     print(f"Lengths: {lengths}")                
    #     print(f"Mask shape: {mask.shape}")        
    #     print(f"Mask example: {mask[0]}")
    #     break


    m=torch.tensor([86.22, 95.07, 117.53, 89.985, 93.626, 95.41], dtype=torch.float32)
    s=torch.tensor([48.275, 49.23, 75.38, 2.81, 8.2, 11.98], dtype=torch.float32)
    par=True
    model1 = QtoRModel(latent_dim=64,num_theta_samples=2, encoder_hidden=128, rotation_hidden=128,theta_isParam=par,theta_mu=m,theta_diagS=s)
    model2 = QtoHModel(temperature=0.05)

    for Q,lengths,mask in dataloader:
        print(Q[0])
        R,B,z = model1(Q,mask)
        H=model2(Q,R,B,z)
        print("H_candidates shape:", H.shape)  # Should be[B, C, N, 3]
        valid_length = lengths[0]
        valid_H_sample0 = H[0, :, :valid_length, :]  # shape: [C, valid_length, 3]
        print("Valid H_candidates for sample 0:\n", valid_H_sample0)
        
                