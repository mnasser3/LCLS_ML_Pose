from QR.QRmodel import *
from QR.utils import *
    
if __name__ == '__main__':
    B = 5
    N = 4
    #data_gen(B,N,3,7,"QH") 
    dataset = DiffractionDataset("data/output_data.npy")
    dataloader = DataLoader(dataset, batch_size=B, collate_fn=collate_fn)

    for padded_Q, lengths, mask in dataloader:
        print(f"Padded Q shape: {padded_Q.shape}")  
        print(f"Lengths: {lengths}")                
        print(f"Mask shape: {mask.shape}")        
        print(f"Mask example: {mask[0]}")
        break


    m=torch.tensor([86.22, 95.07, 117.53, 89.985, 93.626, 95.41], dtype=torch.float32)
    s=torch.tensor([48.275, 49.23, 75.38, 2.81, 8.2, 11.98], dtype=torch.float32)
    par=False
    C=2
    model = QtoRModel(latent_dim=64,num_theta_samples=2, encoder_hidden=128, rotation_hidden=128,theta_isParam=par,theta_mu=m,theta_diagS=s)

    for Q,_,mask in dataloader:
        R,_,_= model(Q,mask)
        print("R_candidates shape:", R.shape)  # Should be[B, C, 3, 3]

        for k in R:
            for i in k:
                if not is_SO3(i):
                    print(is_SO3(i))
                    exit(0)
    print("Example R_candidate:\n", R[0, 0],(is_SO3(R[0,0])))
                