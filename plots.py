import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

# Given parameters
mu = np.log(89)
diag_S = np.log(2)

# Convert Normal parameters to LogNormal parameters
sigma_log = np.sqrt(diag_S)  # Standard deviation
scale = np.exp(mu)  # Scale parameter for LogNormal

# Generate x values
x = np.linspace(0, scale * 2, 1000)

# Compute LogNormal PDF
pdf = lognorm.pdf(x, s=sigma_log, scale=scale)

# Plot the LogNormal distribution
plt.figure(figsize=(8, 5))
plt.plot(x, pdf, label="LogNormal PDF", color="blue")
plt.fill_between(x, pdf, alpha=0.3, color="blue")

# Labels and title
plt.xlabel("Value")
plt.ylabel("Density")
plt.title("LogNormal Distribution Visualization")
plt.legend()
plt.xlim([0, np.percentile(x, 99)])  # Trim extreme values for better visualization

# Show plot
plt.show()
