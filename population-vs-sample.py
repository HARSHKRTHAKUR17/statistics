import numpy as np
import matplotlib.pyplot as plt

# Set the random seed for reproducibility (optional)
np.random.seed(42)

# Step 1: Generate N=1000 random samples from a Poisson distribution with a mean of λ (lambda)
lambda_value = 10  # You can change the lambda value to any positive number
N = 10000
DataPopulation = np.random.poisson(lam=lambda_value, size=N)

# Step 2: Draw a random subset of n=30 samples from DataPopulation
n = 200
DataSample = np.random.choice(DataPopulation, size=n, replace=False)

# Step 3: Compute the mean and standard deviation of both vectors
mean_population = np.mean(DataPopulation)
std_population = np.std(DataPopulation)

mean_sample = np.mean(DataSample)
std_sample = np.std(DataSample)

# Print the results
print("Population Mean:", mean_population)
print("Population Standard Deviation:", std_population)
print("Sample Mean:", mean_sample)
print("Sample Standard Deviation:", std_sample)