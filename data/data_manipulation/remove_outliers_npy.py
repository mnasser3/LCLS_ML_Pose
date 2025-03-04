import numpy as np
import matplotlib.pyplot as plt

data = np.load("data/output_data.npy", allow_pickle=True)
unit_cells = np.array([entry["Unit_Cell"] for entry in data])

means = np.mean(unit_cells, axis=0)
std_devs = np.std(unit_cells, axis=0)

tolerance = 0.3
within_tolerance = np.sum((unit_cells >= (means - tolerance)) & (unit_cells <= (means + tolerance)), axis=0)

labels = ["a", "b", "c", "α", "β", "γ"]
colors = ["black", "red", "blue", "green", "purple", "orange"]

print("Unit Cell Parameter Statistics:")
for i, label in enumerate(labels):
    print(f"{label}: Mean={means[i]:.3f}, Std Dev={std_devs[i]:.3f}")

print("\nCount of unit cells within ±0.3 of the mean:")
for i, label in enumerate(labels):
    print(f"{label}: {within_tolerance[i]}, {within_tolerance[i] / len(unit_cells) * 100:.2f}%")

plt.figure(figsize=(12, 6))
hist_handles = []
for i in range(6):
    hist = plt.hist(unit_cells[:, i], bins=50, alpha=0.6, label=f"{labels[i]}: μ={means[i]:.3f}, σ={std_devs[i]:.3f}", color=colors[i])
    hist_handles.append(hist[2][0])

legend1 = plt.legend()
legend1.get_texts()[0].set_color("black")
legend1.get_texts()[1].set_color("red")

within_tolerance_labels = [f"{labels[i]}: {within_tolerance[i]} ({within_tolerance[i] / len(unit_cells) * 100:.2f}%)" for i in range(6)]
legend2 = plt.legend(hist_handles, within_tolerance_labels, loc="upper left", title="Count within ±0.3")
plt.gca().add_artist(legend1)

plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Distribution of Unit Cell Parameters")
plt.show()
plt.savefig("data/BEFOREunit_cell_parameters_distribution.png")
