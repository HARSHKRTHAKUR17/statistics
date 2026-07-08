import numpy as np
import matplotlib.pyplot as plt

# Set the random seed for reproducibility
np.random.seed(42)

# Generate N=1000 random samples from a Poisson distribution with a mean of λ (lambda)
lambda_value = 10
N = 1000
DataPopulation = np.random.poisson(lam=lambda_value, size=N)

# Parameters for sampling
n = 30    # Sample size
m = 100   # Number of samples to take

# Initialize arrays to store sample means and standard deviations
SampleMeans = np.zeros(m)
SampleSTDs = np.zeros(m)

# Repeat sampling m times
for i in range(m):
    DataSample = np.random.choice(DataPopulation, size=n, replace=False)
    SampleMeans[i] = np.mean(DataSample)
    SampleSTDs[i] = np.std(DataSample)

# Plot a histogram of the SampleMeans
plt.figure(figsize=(10, 6))
plt.hist(SampleMeans, bins=15, color='red', edgecolor='black', alpha=0.7)
plt.title('Histogram of Sample Means')
plt.xlabel('Sample Mean')
plt.ylabel('Frequency')
plt.grid(True)

# Display the histogram
plt.show()