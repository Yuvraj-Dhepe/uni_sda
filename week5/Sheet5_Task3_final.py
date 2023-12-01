## Task 1

import numpy as np
import matplotlib.pyplot as plt 
# Group Members: Dhvaniben Jasoliya, Leutrim Uka, Nicola Horst, Tauqeer Rumaney, Yuvraj Dhepe
# Data
sq_footage = np.array([1500, 1800, 2200, 1200, 1600])
num_bedrooms = np.array([3, 4, 5, 2, 3])
distance_from_city = np.array([2, 1, 0, 3, 2])
price = np.array([200000, 220000, 250000, 180000, 210000])

# Create the design matrix
X = np.column_stack((np.ones_like(sq_footage), sq_footage, num_bedrooms, distance_from_city))
y = price

# Hyperparameter
max_iterations = 3000
epsilon = 1e-6  # Convergence criterion

# Initialize coefficients
np.random.seed(42)  # For reproducibility
beta = np.random.rand(4)

# Perform steepest descent iterations
for iteration in range(max_iterations):
    # Compute residual
    r = y - X @ beta
    
    # Determine the steepest descent direction
    d = X.T @ r
    
    # Compute step size
    alpha = np.dot(r.T, X @ d) / np.dot(X @ d, X @ d)
    
    # Take the step
    beta = beta + alpha * d
    
    # Check the convergence criterion
    if np.linalg.norm(d) < epsilon:
        break

# Print the final coefficients
print("============================")
print("Task 1")
print("Intercept (beta0):", beta[0])
print("Coefficient for sq_footage (beta1):", beta[1])
print("Coefficient for num_bedrooms (beta2):", beta[2])
print("Coefficient for distance_from_city (beta3):", beta[3])


## Task 2
print("============================")
print("Task 2")
# Hyperparameters
max_iterations = 1000
epsilon = 1e-6  # Convergence criterion
learning_rate = 0.01  # Initial learning rate

# Regularization parameter values
lambda_values = np.arange(0, 1.01, 0.01)

# Initialize coefficients
beta_history = np.zeros((len(lambda_values), X.shape[1]))  # Store coefficients for each lambda

# Ridge regression iterations for different lambda values
for idx, lambda_val in enumerate(lambda_values):
    np.random.seed(42)  # For reproducibility
    beta = np.random.rand(X.shape[1]) # Initialize coefficients
    
    for iteration in range(max_iterations):
        # Compute residual
        r = y - X @ beta
        
        # Determine the steepest descent direction with regularization term
        d = X.T @ r + lambda_val * beta
        
        # Compute step size
        alpha = np.dot(r.T, X @ d) / np.dot(X @ d, X @ d)
        
        # Take the step
        beta = beta + alpha * d
        
        # Check the convergence criterion
        if np.linalg.norm(d) < epsilon:
            break
    
    # Save coefficients for the current lambda
    beta_history[idx, :] = beta

# Plot the coefficients as a function of lambda
plt.figure(figsize=(10, 6))
for i in range(X.shape[1]):
    plt.plot(lambda_values, beta_history[:, i], label=f'Coefficient {i}')

plt.xlabel('Lambda')
plt.ylabel('Coefficient Value')
plt.title('Ridge Regression Coefficients vs. Lambda')
plt.legend()
fig_name = "Plot.png"
plt.savefig(fig_name, dpi = 150, bbox_inches = 'tight')

print(f"{fig_name} saved")

