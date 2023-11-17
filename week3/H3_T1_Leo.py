import pandas as pd
import numpy as np

def read_files():
    try:
        X = np.array(pd.read_csv('X.txt', header=None))
        y = np.array(pd.read_csv('Y.txt', header=None))
    except Exception as e:
        print("Data files should be in the same directory as the script!")
        return None, None
    return X, y

def estimate_beta(X: np.array, y: np.array):
    """
    Calculates beta hat: β_hat = (XᵀX)⁻¹Xᵀy.
    
    Args:
        X (np.array): Matrix of dimensions n rows and p features (n x p).
        y (np.array): Vector (array) of n target values
        
    Returns:
        beta_hat (np.array): Vector of estimated weights (beta hat) for linear regression
    """
    
    n = X.shape[0]
    p = X.shape[1]
    
    X_t = X.T  # Xᵀ
    inv_XtX = np.linalg.inv(np.dot(X_t, X))  # (XᵀX)⁻¹
    Xty = np.dot(X_t, y)  # Xᵀy
    
    beta_hat = np.dot(inv_XtX, Xty)  # (XᵀX)⁻¹Xᵀy
    
    return beta_hat

    
def estimate_sigmas(X: np.array, y: np.array, beta_hat: np.array):
    """
    Calculates σ² = εᵀε/n given X, y, and β_hat, with ε = y - X*β_hat.
    
    Args:
        X (np.array): Matrix of dimensions n rows and p features (n x p)
        y (np.array): Vector (array) of n target values
        beta_hat (np.array): Vector of estimated weights (beta hat)
        
    Returns:
        sigma_sq, sigma_sq_ad (tuple): Calculated values of sigma squared and adjusted estimator 
    """
    
    n = X.shape[0]
    p = X.shape[1]
    
    epsilon = y - np.dot(X, beta_hat)
    sigma_sq = np.dot(epsilon.T, epsilon)/n
    
    sigma_sq_ad = np.dot(epsilon.T, epsilon)/(n-p-1)
    
    return (sigma_sq, sigma_sq_ad)

if __name__ == '__main__':
    X, y = read_files()
    
    if X is not None and y is not None:      
        beta_hat = estimate_beta(X, y)
        sigma_sq, sigma_sq_ad = estimate_sigmas(X, y, beta_hat)

        print(f"Beta_hat={beta_hat}")
        print(f"Sigma squared={sigma_sq}")
        print(f"Adjusted estimator={sigma_sq_ad}")    