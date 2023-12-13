import numpy as np
import matplotlib.pyplot as plt


# Set a seed for reproducibility
np.random.seed(42)

# Function to calculate the likelihood ratio
def likelihood_ratio(x, theta_0):
    n = len(x)
    return np.exp(-n * (theta_0 - np.mean(x)))

# Values of theta_0
theta_0_values = [-2, 0, 0.5, 1]

# Generate data points (you need to replace this with your actual data)
data = np.random.exponential(scale=1, size=100)

# Calculate likelihood ratios for each theta_0
likelihood_ratios = [likelihood_ratio(data, theta_0) for theta_0 in theta_0_values]

# Plot the likelihood ratios
plt.plot(theta_0_values, likelihood_ratios, marker='o')
plt.xlabel('Theta_0')
plt.ylabel('Likelihood Ratio')
plt.title('Likelihood Ratio Test Statistic')

# Save the plot as a PNG file
plt.savefig('task_2_likelihood_ratio_plot.png', dpi = 150, bbox_inches = 'tight')
