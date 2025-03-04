import numpy as np

# Load saved .npy file
data = np.load("data/output_data.npy", allow_pickle=True)


sample = data[0]
print("\n##############################################")
print("Sample values for an image + R + unit cell params")
for key, value in sample.items():
    if isinstance(value, np.ndarray) and isinstance(value[0], np.ndarray):  # Check for nested lists
        print(f"\n{key}: Shape={len(value)}x{len(value[0])} (Preview={value[:2]})")
        print()
    else:  # Scalar values (e.g., File name, Unit_Cell)
        print(f"{key}: {value}")
        print()
print("##############################################")
print("dataset:",type(data))  # Should be a NumPy array of dicts
print()
print("Q:", type(data[0]["Q"]),data[0]["Q"].shape) 
print()
print("Miller indices:", type(data[0]["Miller_Indices"]),data[0]["Miller_Indices"].shape) 
print()
print("R", type(data[0]["U"]),data[0]["U"].shape)
print()
print("Unit Cell", type(data[0]["Unit_Cell"]))
print()
print("Each sample has keys:",data[0].keys())  


 
