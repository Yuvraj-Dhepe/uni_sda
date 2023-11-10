import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib
import matplotlib.pyplot as plt
# Group Members: Dhvaniben Jasoliya, Leutrim Uka, Nicola Horst, Tauqeer Rumaney, Yuvraj Dhepe
# since there seems to be an issue
matplotlib.use('TkAgg')


def generate_scatter_plot(df: DataFrame, x: str, y: str, title: str, save_dir=None):
    """
    Generate a Scatter Plot from a dataframe
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_title(title)
    ax.set_xlabel("Amount of Berries")
    ax.set_ylabel("Amount of Wine")
    ax.set_title("Amount of Produced Wine vs. Amount of Berries per 100m2")
    plt.scatter(x=df[x], y=df[y])

    if save_dir:
        plt.savefig(save_dir)


def compute_linear_regression(data: DataFrame):
    """
    Compute a linear regression to predict the amount of wine that is produced by the amount of berries
    """
    # pos 0 = Wine pos 1 = Cluster
    as_matrix: np.array = np.array(data.values, 'float')

    # retrieve the x (Cluster) and y (Wine) values and reshape to for beta 0, beta 1 shape
    y = as_matrix[:, 0]

    # reshape x to be in the format of [beta 0, beta 1] which is the linear function
    # we set the first value to be always 2 while the other one will be x so when multiplying beta to x
    # we have beta 0 * 1 + beta 1 * x
    x = as_matrix[:, 1].reshape([12, 1])
    x = np.hstack([np.ones_like(x), x])

    # compute the optimal theta
    theta: np.array = analytic_solution(x, y)

    return x, y, theta


def analytic_solution(x, y):
    """
    Compute the analytical solution for a simple linear regression derived in the lecture
    (X^TX)^-1Xy
    """
    return np.dot(np.dot(np.linalg.inv(np.dot(x.T, x)), x.T), y)


def plot_linear_regression(X, x, y, theta, save_dir=None):
    """
    Plot the linear regression line among the scatter plot of samples
    """
    # plot linear fit for our theta
    plt.scatter(X, y)
    plt.plot(X, x @ theta, '-')
    plt.ylabel('Wine Production')
    plt.xlabel('Amount of Berries')
    plt.legend(['Wine Prod', 'LinearFit'])
    plt.title('Amount Berries vs Wine Production and Linear Fit')

    if save_dir:
        plt.savefig(save_dir)


if __name__ == "__main__":
    # Please Check that the data-file path is correct
    file_path: str = r'./wine.txt'

    # Plots will be saved in your current directory
    save_dir: str = r"./scatter_plot"
    save_dir_lr: str = r"./linear_fit"

    # Load the data
    wine_df: DataFrame = pd.read_csv(file_path, sep=' ', header=0, names=['Year', 'Wine', 'Cluster'])

    # Task 3.1 Generate Scatter Plot
    print("TASK 1 Scatter-Plot:")
    generate_scatter_plot(wine_df, x='Cluster', y='Wine', title='default', save_dir=save_dir)
    print(f"Scatter Plot has been saved in {save_dir}\n\n")

    # Task 3.2 and 3.3
    print("TASK 3.2 and 3.3 Fit Linear Model and Plot it:")
    x, y, theta = compute_linear_regression(wine_df.drop("Year", axis=1))
    print(f"Beta 0: {theta[0]}\nBeta 1: {theta[1]}")
    plot_linear_regression(wine_df["Cluster"], x, y, theta, save_dir_lr)
    print(f"\nLinear Regression has been saved in {save_dir_lr}\n\n")

    # Task 3.4
    cluster: int = 100
    prediction = theta[0] + (100 * theta[1])
    print("TASK 3.4 Prediction:")
    print(f'for a cluster size of {cluster}, the predicted amount of wine produced is {prediction}')
