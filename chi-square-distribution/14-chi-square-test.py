import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, chi2

# Data
n = np.arange(18)  # Number of counts per 10-second interval

Observed = np.array([
    1, 6, 11, 28, 56, 105, 126, 146,
    164, 161, 123, 101, 74, 53,
    23, 15, 9, 5
])

# --------------------------------------------------
# Combine first three bins (0, 1, and 2 counts)
# --------------------------------------------------

Observed_combined = np.copy(Observed)
Observed_combined[2] = np.sum(Observed[0:3])   # Sum first three bins
Observed_combined = np.delete(Observed_combined, [0, 1])

# Adjust x-values to match combined bins
n_combined = np.copy(n)
n_combined[2] = 2
n_combined = np.delete(n_combined, [0, 1])

# --------------------------------------------------
# Estimate lambda
# --------------------------------------------------

sample_mean = np.average(n, weights=Observed)

# Total observations
total_counts = np.sum(Observed)

# Expected frequencies
poisson_probs = poisson.pmf(n, mu=sample_mean)
Expected = poisson_probs * total_counts

# Combine first three expected bins
Expected_combined = np.copy(Expected)
Expected_combined[2] = np.sum(Expected[0:3])
Expected_combined = np.delete(Expected_combined, [0, 1])

# --------------------------------------------------
# Chi-Squared Goodness-of-Fit Test
# --------------------------------------------------

chi_squared_stat = np.sum(
    (Observed_combined - Expected_combined) ** 2
    / Expected_combined
)

# Number of bins - 1 - number of estimated parameters
degrees_of_freedom = len(Observed_combined) - 1 - 1

p_value = 1 - chi2.cdf(
    chi_squared_stat,
    df=degrees_of_freedom
)

print(f"Chi-Squared Statistic: {chi_squared_stat:.4f}")
print(f"Degrees of Freedom: {degrees_of_freedom}")
print(f"P-value: {p_value:.4f}")

# --------------------------------------------------
# Plot Chi-Squared PDF
# --------------------------------------------------

x = np.linspace(0, 40, 1000)
pdf = chi2.pdf(x, df=degrees_of_freedom)

critical_value = chi2.ppf(
    0.95,
    df=degrees_of_freedom
)

plt.figure(figsize=(10, 6))

plt.plot(
    x,
    pdf,
    "b-",
    lw=2,
    label=f"Chi-Squared PDF (df={degrees_of_freedom})"
)

# Rejection region
plt.fill_between(
    x,
    pdf,
    where=(x >= critical_value),
    color="red",
    alpha=0.3,
    label="Rejection Region (α=0.05)"
)

# Test statistic
plt.axvline(
    x=chi_squared_stat,
    color="green",
    linestyle="--",
    label=f"Chi-Squared Stat = {chi_squared_stat:.2f}"
)

# Critical value
plt.axvline(
    x=critical_value,
    color="red",
    linestyle="--",
    label=f"Critical Value = {critical_value:.2f}"
)

plt.title(
    "Chi-Squared Test Probability Density Function",
    fontsize=14
)

plt.xlabel("Chi-Squared Value", fontsize=12)
plt.ylabel("PDF", fontsize=12)

plt.legend()
plt.grid(True)

plt.show()

# --------------------------------------------------
# Decision
# --------------------------------------------------

if chi_squared_stat > critical_value:
    print(
        "Reject the null hypothesis: "
        "The data is not consistent with a Poisson distribution."
    )
else:
    print(
        "Fail to reject the null hypothesis: "
        "The data is consistent with a Poisson distribution."
    )