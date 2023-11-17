import pandas as pd
import numpy as np
# Group Members: Dhvaniben Jasoliya, Leutrim Uka, Nicola Horst, Tauqeer Rumaney, Yuvraj Dhepe
# Load X and Y data from CSV files
X = pd.read_csv('X.txt', header=None)
Y = pd.read_csv('Y.txt', header=None)

# Add a column of ones to X for the intercept term
X.insert(0, 'Intercept', 1)

# Convert data to numpy arrays
X = X.to_numpy()
Y = Y.to_numpy().flatten()  # Assuming Y is a column vector

# Compute ML estimator for beta
beta_hat = np.linalg.inv(X.T @ X) @ X.T @ Y

# Compute residuals
residuals = Y - X @ beta_hat

# Compute ML estimator for sigma^2
sigma2_ML = np.sum(residuals**2) / (X.shape[0] - X.shape[1])

# Compute adjusted sigma^2 (assuming unbiased estimator)
sigma2_ad = sigma2_ML * (X.shape[0] / (X.shape[0] - X.shape[1]))

# Print results
print("ML Estimators for Beta:", beta_hat)
print("ML Estimator for sigma^2:", sigma2_ML)
print("Adjusted sigma^2:", sigma2_ad)