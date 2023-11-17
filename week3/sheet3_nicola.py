import numpy as np
import pandas as pd
from pandas import DataFrame


def read_data(path_x: str, path_y: str):
    x: DataFrame = pd.read_csv(path_x, sep=',', header=None, names=['x1', 'x2', 'x3'])
    y: DataFrame = pd.read_csv(path_y, sep=' ',  header=None, names=['y'])

    # transform the df into numpy arrays
    _x = np.array(x.values, 'float')
    _y = np.array(y.values, 'float')

    # add ones in x[0: ] for beta 0
    _x = np.column_stack((np.ones(len(_x)), _x))
    return _x, _y


def get_beta_hat_estimate(x: np.array, y: np.array):
    """
    We know that minimizing the LS estimator is equivalent to maximising the (log) likelihood estimate ML. Therfore
    to obtain the ML w.r.t beta, we compute the optimum for the LS estimator by using the derived formula
    (X.T @ X)^-1 @ X.T @ y
    """
    beta_hat_estimate = np.linalg.inv(x.T @ x) @ x.T @ y
    return beta_hat_estimate


def get_sigma_hat_estimate(x: np.array, y: np.array, beta_hat: np.array):
    """
    Using the ML Estimator for beta (beta_hat) we can compute the estimate for the squared variance
    Therefore we compute the sum of squared residuals with is equivalent to eps.T @ eps since eps = (y - X @ beta_hat)
    finally we return the sse divided by the number of samples to obtain our ML estimate for sigma^2 (sigma^2_hat)
    """
    sse: np.array = np.sum((y - x @ beta_hat)**2)
    n = len(x)
    return sse / n


def get_adjusted_sigma_hat_estimate(x: np.array, y: np.array, beta_hat: np.array):
    """
    Using the ML Estimator for beta (beta_hat) we can compute the adjusted estimator for the squared variance.
    This is almost the same computation as the non-adjusted version but we divide not by the number of samples but
    n-p-1 where n - 1 the total amount of freedom coming from our samples.
    p = is the amount of freedom we get form our prediction variables beta
    """
    sse: np.array = np.sum((y - x @ beta_hat) ** 2)
    n = len(x)
    p = len(beta_hat)

    return sse / (n - p - 1)


if __name__ == "__main__":
    path_x: str = "./X.txt"
    path_y: str = "./Y.txt"

    # load the data
    x, y = read_data(path_x, path_y)

    # compute the estimator
    beta_hat_estimator = get_beta_hat_estimate(x, y)
    sigma_hat_estimator = get_sigma_hat_estimate(x, y, beta_hat_estimator)
    ad_sigma_hat_sse = get_adjusted_sigma_hat_estimate(x, y, beta_hat_estimator)

    print(f"ML Estimator β hat: \n{beta_hat_estimator}")
    print(f"ML Estimator of σ^2: {sigma_hat_estimator}")
    print(f"ML Estimator of adjusted σ^2 sse: {ad_sigma_hat_sse}")
