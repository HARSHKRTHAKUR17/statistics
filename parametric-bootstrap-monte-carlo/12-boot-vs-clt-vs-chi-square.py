# USE BOOTSTRAP TO ESTIMATE DISTRIBUTION OF VARIANCE FOR NORMAL DIST

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import chi2

# Set random seed for reproducibility
np.random.seed(42)

# Generate data from a normal distribution
mu = 10          # True mean of the normal distribution
sigma = 2        # True standard deviation
N = 100          # Number of samples

data = np.random.normal(mu, sigma, N)

# Method of Moments estimates for mean and variance
estimated_mu = np.mean(data)
estimated_variance = np.var(data, ddof=1)

# Number of synthetic datasets to generate (bootstrapping)
num_synthetic_samples = 1000
synthetic_variances = []

# Generate synthetic datasets using the estimated mean and variance,
# and compute sample variances
for _ in range(num_synthetic_samples):

    synthetic_data = np.random.normal(
        loc=estimated_mu,
        scale=np.sqrt(estimated_variance),
        size=N
    )

    synthetic_variance = np.var(
        synthetic_data,
        ddof=1
    )

    synthetic_variances.append(synthetic_variance)

# ---------------------------------------------------------
# CLT approximation for the variance estimator
# ---------------------------------------------------------

# Standard error of the sample variance (Normal distribution)
clt_std_error = estimated_variance * np.sqrt(2 / (N - 1))

# Histogram
plt.figure(figsize=(10, 6))

plt.hist(
    synthetic_variances,
    bins=30,
    color="gray",
    edgecolor="white",
    alpha=0.7,
    density=True,
    label="Synthetic Variance Distribution"
)

# Generate the CLT approximation
x = np.linspace(
    min(synthetic_variances),
    max(synthetic_variances),
    1000
)

normal_approximation = norm.pdf(
    x,
    loc=estimated_variance,
    scale=clt_std_error
)

# Plot CLT approximation
plt.plot(
    x,
    normal_approximation,
    "y-",
    lw=2,
    label=f"Normal Approximation (CLT)\n(mean={estimated_variance:.2f}, std={clt_std_error:.4f})"
)

# ---------------------------------------------------------
# Exact Chi-square distribution
# ---------------------------------------------------------

df = N - 1

chi2_x = np.linspace(
    min(synthetic_variances),
    max(synthetic_variances),
    1000
)

chi2_pdf = (
    chi2.pdf(df * chi2_x / estimated_variance, df)
    * (df / estimated_variance)
)

plt.plot(
    chi2_x,
    chi2_pdf,
    "c-",
    lw=2,
    label=f"Chi-Square Fit (df={df})"
)

# Labels
plt.title(
    "Synthetic Distribution of Sample Variance with Normal Appx. and Chi-Square Fit",
    fontsize=14
)

plt.xlabel("Sample Variance", fontsize=12)
plt.ylabel("Density", fontsize=12)

plt.legend()
plt.grid(True)

plt.show()

# ---------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------

synthetic_mean_variance = np.mean(synthetic_variances)
synthetic_std_variance = np.std(synthetic_variances)

print(f"Mean of Synthetic Sample Variance: {synthetic_mean_variance:.4f}")
print(f"Std Dev of Synthetic Sample Variance: {synthetic_std_variance:.4f}")
print(f"Method of Moments Estimate of Variance: {estimated_variance:.4f}")