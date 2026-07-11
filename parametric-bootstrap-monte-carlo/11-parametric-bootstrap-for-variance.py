# USE BOOTSTRAP TO ESTIMATE DISTRIBUTION OF VARIANCE FOR NORMAL DIST

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Set random seed for reproducibility
np.random.seed(42)

# Generate data from a normal distribution
mu = 10        # True mean of the normal distribution
sigma = 2      # True standard deviation of the normal distribution
N = 100        # Number of samples

data = np.random.normal(mu, sigma, N)

# Method of Moments estimates for mean and variance
estimated_mu = np.mean(data)
estimated_variance = np.var(data, ddof=1)   # Unbiased estimate

# Number of synthetic datasets to generate (bootstrapping)
num_synthetic_samples = 1000
synthetic_variances = []

# Generate synthetic datasets using the estimated mean and variance,
# and compute their sample variances
for _ in range(num_synthetic_samples):

    # Generate synthetic dataset
    synthetic_data = np.random.normal(
        loc=estimated_mu,
        scale=np.sqrt(estimated_variance),
        size=N
    )

    # Compute sample variance
    synthetic_variance = np.var(
        synthetic_data,
        ddof=1
    )

    synthetic_variances.append(synthetic_variance)

# Plot the distribution of the synthetic sample variances
plt.figure(figsize=(10, 6))

plt.hist(
    synthetic_variances,
    bins=30,
    color="blue",
    edgecolor="black",
    alpha=0.7,
    density=True,
    label="Synthetic Variance Distribution"
)

# Add labels, title, and legend
plt.title(
    "Bootstrap (Synthetic) Distribution of Sample Variance",
    fontsize=14
)

plt.xlabel("Sample Variance", fontsize=12)
plt.ylabel("Density", fontsize=12)

plt.grid(True)
plt.legend()

plt.show()

# Print summary statistics
print(f"Mean of Synthetic Sample Variance: {np.mean(synthetic_variances):.4f}")
print(f"Std Dev of Synthetic Sample Variance: {np.std(synthetic_variances):.4f}")
print(f"Method of Moments Estimate of Variance: {estimated_variance:.4f}")