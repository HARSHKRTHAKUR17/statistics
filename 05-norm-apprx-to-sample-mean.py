import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Set the random seed for reproducibility
# np.random.seed(42)

# Generate N=1000 random samples from a Poisson distribution with a mean of λ (lambda)
lambda_value = 10
N = 10000
DataPopulation = np.random.poisson(lam=lambda_value, size=N)

# Parameters for sampling
n = 30   # Sample size # we can increase this for more accurate results
m = 100  # Number of samples to take # we can increase this too for better results

# Initialize arrays to store sample means and standard deviations
SampleMeans = np.zeros(m)
SampleSTDs = np.zeros(m)

# Repeat sampling m times
for i in range(m):
    DataSample = np.random.choice(DataPopulation, size=n, replace=False)
    SampleMeans[i] = np.mean(DataSample)
    SampleSTDs[i] = np.std(DataSample)

# Calculate the population mean and standard deviation
mean_population = np.mean(DataPopulation)
std_population = np.std(DataPopulation)

# Compute the standard deviation of the sample means (Standard Error of the Mean)
std_error = std_population / np.sqrt(n)

# Generate the normal distribution that approximates the distribution of sample means
x = np.linspace(min(SampleMeans), max(SampleMeans), 100)
normal_approximation = norm.pdf(
    x,
    loc=mean_population,
    scale=std_error
)

# Plot the histogram of the SampleMeans
plt.figure(figsize=(10, 6))
plt.hist(
    SampleMeans,
    bins=15,
    color='red',
    edgecolor='black',
    alpha=0.7,
    density=True
)

# Plot the normal distribution approximation on top of the histogram
plt.plot(
    x,
    normal_approximation,
    'b-',
    label=f'Normal Approximation\n(mean={mean_population:.2f}, std={std_error:.2f})'
)

# Label the axes and add a title
plt.xlabel('Sample Mean')
plt.ylabel('Density')
plt.title('Histogram of Sample Means with Normal Approximation')

# Add a legend
plt.legend()

# Display the plot
plt.show()