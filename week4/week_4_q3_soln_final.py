import numpy as np

if __name__ == "__main__":
    distances = np.array([2, 4, 6, 8, 10])
    pollution = np.array([11.5, 10.2, 10.3, 9.68, 9.32])

    # Given data
    distances = np.array([2, 4, 6, 8, 10])
    pollution = np.array([11.5, 10.2, 10.3, 9.68, 9.32])
    
    # Calculate means
    x_bar = np.mean(distances)
    y_bar = np.mean(pollution)
    
    # Calculate least squares estimators
    beta_1 = np.sum((distances - x_bar) * (pollution - y_bar)) / np.sum((distances - x_bar)**2)
    beta_0 = y_bar - beta_1 * x_bar
    
    # Print results
    print("Least Squares Estimators:")
    print("Beta_0:", beta_0)
    print("Beta_1:", beta_1)

    # Calculate residuals
    residuals = pollution - (beta_0 + beta_1 * distances)
    
    # Calculate adjusted MLE of the variance
    sigma_squared_hat = np.sum(residuals**2) / (len(distances) - 2)
    
    # Print result
    print("Adjusted MLE of Variance:")
    print("Sigma^2_hat:", sigma_squared_hat)