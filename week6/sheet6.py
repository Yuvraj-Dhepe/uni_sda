import numpy as np


def get_z_0(m: int):
    return np.random.normal(loc=0, scale=0.5, size=m)


def get_eps(m):
    return np.random.normal(0, 0.5, size=(m, 1))


def evolution(z_n):
    return 0.99 * z_n + get_eps(m=len(z_n))


def get_observations(z_n):
    return z_n + get_eps(1)


def mse(y, _y):
    return (y - _y) ** 2


def mse_kalman(kalman_means, true_signal):
    sse = 0
    for i, point in enumerate(true_signal):
        sse += mse(point, kalman_means[i])

    return sse / len(true_signal)


def mse_enkf(enkf_means, true_signal):
    sse = 0
    for i, point in enumerate(true_signal):
        sse += mse(point, enkf_means[i].mean())

    return sse / len(true_signal)


# Kalman Filter
def kalman_filter(initial_mean, initial_cov, evolution_noise, evolution_coef, observations, num_steps):
    # Initialize the means and covariances over time for plotting
    state_means = np.zeros(num_steps)
    state_covariances = np.zeros(num_steps)

    # Initialization of the mean and the variances
    state_mean = initial_mean
    state_covariance = initial_cov

    for step in range(num_steps):
        # 1. produce prediction
        predicted_state_mean = evolution_coef * state_mean
        predicted_state_covariance = evolution_coef ** 2 * state_covariance + evolution_noise ** 2

        # 2 update the model by using the new observation to compute the kalman gain
        k_gain = predicted_state_covariance / (predicted_state_covariance + evolution_noise ** 2)
        state_mean = predicted_state_mean + (k_gain * (observations[step] - predicted_state_mean))
        state_covariance = predicted_state_covariance - (k_gain * predicted_state_covariance)

        state_means[step] = state_mean
        state_covariances[step] = state_covariance

    return state_means, state_covariances


# Ensemble Kalman Filter
def ensemble_kalman_filter(obs, initial_ensemble, num_steps):
    ensemble_size = initial_ensemble.shape[0]
    state_means = np.zeros((num_steps, ensemble_size))

    # Initialization
    ensemble = initial_ensemble

    for step in range(num_steps):
        # Prediction step
        ensemble = evolution(ensemble)

        # Update step
        kalman_gain = np.cov(ensemble, rowvar=False) / (np.cov(ensemble, rowvar=False) + 0.5 ** 2)
        ensemble = ensemble + kalman_gain * (obs[step] - ensemble)

        # compute the mean for the ensemble
        state_means[step, :] = np.mean(ensemble, axis=1)

    return state_means


if __name__ == "__main__":
    data_path: str = "./Data/reference_signal.txt"

    # load the true signal
    signal: list = []
    with open(data_path) as reference_txt:
        for line in reference_txt:
            signal.append(float(line))

    # Define the number of steps
    num_steps: int = len(signal)

    # Generate true signal and observations
    true_signal = np.array(signal)
    observations = np.zeros(num_steps)

    # Initialization
    observations[0] = true_signal[0] + np.random.normal(0, 0.5)

    # Generate true signal and observations
    for step in range(1, num_steps):
        observations[step] = get_observations(true_signal[step])

    # Kalman Filter
    kf_means, _ = kalman_filter(observations=observations, initial_mean=0, initial_cov=0.5,evolution_coef=0.99, evolution_noise=0.5, num_steps=num_steps)

    # Ensemble Kalman Filter with different ensemble sizes
    ensemble_sizes = [5, 10, 25, 50]
    mse_kal = mse_kalman(kalman_means=kf_means, true_signal=true_signal)

    mse_enkfs = []
    for m in ensemble_sizes:
        initial_ensemble = np.random.normal(0, 0.5, size=(m, 1))
        enkf_means = ensemble_kalman_filter(observations, initial_ensemble, num_steps=num_steps)
        mse_enkfs.append(mse_enkf(enkf_means=enkf_means, true_signal=true_signal))

    print(f"Mean Squared Error of Kalman Filter: {mse_kal}")
    for ensemble_size, mse_enkf in zip(ensemble_sizes, mse_enkfs):
        print(f"Mean Squared Error of Ensemble Kalman Filter with Ensemble Size {ensemble_size}: {mse_enkf}")