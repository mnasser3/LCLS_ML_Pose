import numpy as np

# Load dataset
data = np.load("data/output_data.npy", allow_pickle=True)

# Extract all Q matrices
Q_matrices = [sample["Q"] for sample in data]

# Check if all Q matrices have the same shape
shapes = [q.shape for q in Q_matrices]
unique_shapes = set(shapes)

Q_all = np.concatenate(Q_matrices, axis=0)
Q_mean = np.mean(Q_all, axis=0)
Q_std = np.std(Q_all, axis=0)

print("Mean of Qx, Qy, Qz:", Q_mean)
print("Standard deviation of Qx, Qy, Qz ):", Q_std)
