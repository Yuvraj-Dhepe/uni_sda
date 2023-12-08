import numpy as np
#Group members:
#Dhvaniben Jasoliya
#Leutrim Uka
#Nicola Horst
#Taqueer Rumaney
#Yuvraj Dhepe

def get_eps(m):
    return np.random.normal(0, 0.5, size=(m, 1))


def evolution(z_n):
    return 0.99 * z_n + get_eps(m=len(z_n))


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
    state_mean = initial_mean  # m_0
    state_covariance = initial_cov  # C_0

    for step in range(num_steps):
        # 1. produce prediction
        predicted_state_mean = evolution_coef * state_mean  # M^-1 0.99
        predicted_state_covariance = evolution_coef ** 2 * state_covariance + evolution_noise ** 2

        # 2 update the model by using the new observation to compute the kalman gain
        k_gain = predicted_state_covariance / (predicted_state_covariance + evolution_noise ** 2)
        state_mean = predicted_state_mean - (k_gain * (predicted_state_mean - observations[step]))
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

        ensemble = ensemble - kalman_gain * (ensemble - obs[step])

        # compute the mean for the ensemble
        state_means[step, :] = np.mean(ensemble, axis=1)

    return state_means


if __name__ == "__main__":
    data_path: str = "../reference_signal.txt"
    obs_path: str = "../data.txt"

    # load the true signal
    signal: list = []
    observations: list = []
    with open(data_path) as reference_txt:
        for line in reference_txt:
            signal.append(float(line))

    with open(data_path) as reference_txt:
        for line in reference_txt:
            observations.append(float(line))

    # Define the number of steps
    num_steps: int = len(signal)

    # Generate true signal and observations
    true_signal = np.array(signal)
    observations = np.array(observations)

    # Kalman Filter
    kf_means, _ = kalman_filter(observations=observations, initial_mean=0, initial_cov=0.5, evolution_coef=0.99, evolution_noise=0.5, num_steps=num_steps)

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


    print(
        "\n\n"
        "Comments:\n"
        "It Appears, that the Kalman Filter performs a bit better than the Ensemble Kalman Filter. \n"
        "This can have different reasons. One would be the simplicity of the data. Ensemble Kalman Filters are mostly \n"
        "used when the data has high dimensionality (e.g. 10e8). However in this case the dimensionality of the data is 1. \n"
        "Another reason could be that the number of Ensemble members is rather small. It is observable that the MSE \n"
        "for an increasing number of members is becoming smaller. \n"
        "So increasing the number of ensemble members does improve the performance. \n"
        "At the end it is hard to say why exactly the performance is worse but it will be a combination of the above \n"
        "mentioned facts"
    )