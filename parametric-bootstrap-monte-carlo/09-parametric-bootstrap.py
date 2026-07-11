import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# Data
n = np.arange(18)  # Number of counts per 10-second interval

Observed = [
    1, 6, 11, 28, 56, 105, 126, 146,
    164, 161, 123, 101, 74, 53,
    23, 15, 9, 5
]

# Calculate the sample mean (initial estimate of lambda)
sample_mean = np.average(n, weights=Observed)

# Number of bootstrap samples
num_bootstrap = 1000
bootstrap_lambdas = []

# Bootstrap procedure
for _ in range(num_bootstrap):

    # Generate a synthetic dataset using the estimated lambda
    synthetic_data = np.random.poisson(
        lam=sample_mean,
        size=np.sum(Observed)
    )

    # Bin the synthetic data to match the observed format
    synthetic_hist, _ = np.histogram(
        synthetic_data,
        bins=np.arange(19)
    )

    # Calculate the new lambda estimate from the synthetic data
    new_lambda = np.average(
        n,
        weights=synthetic_hist
    )

    # Store the estimate
    bootstrap_lambdas.append(new_lambda)

# Plot the distribution of bootstrap lambda estimates
plt.figure(figsize=(10, 6))

plt.hist(
    bootstrap_lambdas,
    bins=30,
    color="green",
    edgecolor="black",
    alpha=0.7
)

plt.title(
    "Bootstrap Distribution of Lambda Estimates",
    fontsize=14
)

plt.xlabel("Lambda Estimate", fontsize=12)
plt.ylabel("Frequency", fontsize=12)

plt.grid(True)

# Print the mean and standard deviation of bootstrap estimates
bootstrap_mean = np.mean(bootstrap_lambdas)
bootstrap_std = np.std(bootstrap_lambdas)

print(f"Bootstrap Mean of Lambda: {bootstrap_mean:.4f}")
print(f"Bootstrap Std Dev of Lambda: {bootstrap_std:.4f}")

plt.show()