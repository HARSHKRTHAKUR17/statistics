import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ---------------------------------------------------
# Set seed for reproducibility
# ---------------------------------------------------
np.random.seed(45)

# ---------------------------------------------------
# True parameters
# ---------------------------------------------------
true_a = 2 / 3
true_b = 0

sigma_noise = 1      # Standard deviation of Gaussian noise

# ---------------------------------------------------
# Generate synthetic data
# ---------------------------------------------------
n_points = 20

x = np.random.uniform(0, 5, n_points)
noise = np.random.normal(0, sigma_noise, n_points)

y = true_a * x + true_b + noise

# Add an outlier
y[19] = y[19] + 3

# ---------------------------------------------------
# Plot generated data
# ---------------------------------------------------
plt.figure(figsize=(8, 5))

plt.scatter(
    x,
    y,
    label="Data points",
    color="blue",
)

plt.plot(
    x,
    true_a * x + true_b,
    label="True line: y = 2/3 x",
    color="green",
)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Generated Data with Noise")
plt.legend()
plt.grid(True)

plt.show()

# ===================================================
# Least Squares Estimation
# ===================================================

A = np.vstack([x, np.ones(len(x))]).T

theta_lsq, _, _, _ = np.linalg.lstsq(
    A,
    y,
    rcond=None,
)

# ===================================================
# Maximum Likelihood Estimation (MLE)
# ===================================================

def negative_log_likelihood(theta):
    a, b = theta

    y_pred = a * x + b

    residuals = y - y_pred

    return np.sum(residuals ** 2) / (2 * sigma_noise ** 2)


theta_mle = minimize(
    negative_log_likelihood,
    [1, 1],
).x

# ===================================================
# Bayesian MAP Estimation
# Prior:
# b ~ N(0, sigma_prior_b²)
# ===================================================

sigma_prior_b = 0.01


def negative_log_posterior(theta):
    a, b = theta

    log_prior_b = -0.5 * (b ** 2 / sigma_prior_b ** 2)

    return negative_log_likelihood(theta) - log_prior_b


theta_map = minimize(
    negative_log_posterior,
    [1, 1],
).x

# ===================================================
# Results
# ===================================================

print(f"True parameters:")
print(f"  a = {true_a:.3f}, b = {true_b:.3f}")

print(f"\nLeast Squares:")
print(f"  a = {theta_lsq[0]:.3f}, b = {theta_lsq[1]:.3f}")

print(f"\nMaximum Likelihood:")
print(f"  a = {theta_mle[0]:.3f}, b = {theta_mle[1]:.3f}")

print(f"\nMAP Estimate:")
print(f"  a = {theta_map[0]:.3f}, b = {theta_map[1]:.3f}")

# ===================================================
# Plot all fitted lines
# ===================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    x,
    y,
    label="Data points",
    color="blue",
)

plt.plot(
    x,
    true_a * x + true_b,
    label="True line",
    color="green",
    linewidth=2,
)

plt.plot(
    x,
    theta_lsq[0] * x + theta_lsq[1],
    label="Least Squares",
    color="red",
)

plt.plot(
    x,
    theta_mle[0] * x + theta_mle[1],
    label="MLE",
    color="orange",
)

plt.plot(
    x,
    theta_map[0] * x + theta_map[1],
    label="MAP",
    color="purple",
)

plt.xlabel("x")
plt.ylabel("y")

plt.title("Comparison of Estimations")

plt.legend()
plt.grid(True)

plt.show()