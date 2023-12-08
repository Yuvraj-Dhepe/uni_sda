import numpy as np
#Group members:
#Dhvaniben Jasoliya
#Leutrim Uka
#Nicola Horst
#Taqueer Rumaney
#Yuvraj Dhepe
# For maximum spacing estimator we have to maximize the geometric mean of sample spacing i.e. product of difference of cdfs for samples x1 to xn
# Load the data sets from Moodle (replace with your actual data loading)
data_set_1 = np.loadtxt('sampleset_1_problemsheet4_ex1.txt')
data_set_2 = np.loadtxt('sampleset_2_problemsheet4_ex1.txt')
data_set_3 = np.loadtxt('sampleset_3_problemsheet4_ex1.txt')

# Calculate Maximum Spacing Estimator for each set
n1 = len(data_set_1)
theta_mse_1 = (n1 + 1) / n1 * np.max(data_set_1)

n2 = len(data_set_2)
theta_mse_2 = (n2 + 1) / n2 * np.max(data_set_2)

n3 = len(data_set_3)
theta_mse_3 = (n3 + 1) / n3 * np.max(data_set_3)

# Print the results
print(f"Maximum Spacing Estimator for Set 1: {theta_mse_1}")
print(f"Maximum Spacing Estimator for Set 2: {theta_mse_2}")
print(f"Maximum Spacing Estimator for Set 3: {theta_mse_3}")
