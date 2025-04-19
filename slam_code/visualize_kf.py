import numpy as np
import matplotlib.pyplot as plt
import os

# Load the predicted and corrected position logs
predicted_path = "vslam_final_results/predicted_positions.npy"
corrected_path = "vslam_final_results/corrected_positions.npy"

if not os.path.exists(predicted_path) or not os.path.exists(corrected_path):
    raise FileNotFoundError("Make sure both .npy files are in vslam_final_results/")

# Load data
predicted = np.load(predicted_path)
corrected = np.load(corrected_path)

# Check shape
assert predicted.shape == corrected.shape, "Shape mismatch between predicted and corrected arrays"

# Plot X, Y, Z axes
plt.figure(figsize=(12, 6))
axes = ['X', 'Y', 'Z']
colors = ['red', 'green', 'blue']

for i in range(3):
    plt.plot(predicted[:, i], linestyle='--', color=colors[i], label=f'Predicted {axes[i]}')
    plt.plot(corrected[:, i], linestyle='-', color=colors[i], label=f'Corrected {axes[i]}')

plt.title('Kalman Filter - Predicted vs Corrected Position (X, Y, Z)')
plt.xlabel('Frame')
plt.ylabel('Position (meters)')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save and show
plt.savefig("kalman_position_comparison.png", dpi=300)
plt.show()
