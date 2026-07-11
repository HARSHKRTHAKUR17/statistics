import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binomtest

# Set the random seed for reproducibility
np.random.seed(41)

# Function to compute the p-value for the hypothesis that the coin is fair
def compute_p_value(flips):
    heads = np.sum(flips)      # Number of heads
    n_flips = len(flips)       # Total number of flips

    # Two-sided exact binomial test
    p_value = binomtest(
        heads,
        n=n_flips,
        p=0.5,
        alternative="two-sided"
    ).pvalue

    return p_value


# ----------------------------
# Simulation parameters
# ----------------------------

# Start with 20 coin flips
initial_n = 20

# Add 30 more flips one at a time
additional_n = 30

# Final number of flips
total_n = initial_n + additional_n


# ----------------------------
# Generate initial flips
# ----------------------------

# 1 = Head, 0 = Tail
flips = np.random.binomial(
    n=1,
    p=0.5,
    size=initial_n
)


# ----------------------------
# Store results
# ----------------------------

n_values = list(range(initial_n, total_n + 1))
p_values = []


# ----------------------------
# Sequentially add flips
# ----------------------------

for _ in range(additional_n + 1):

    # Compute current p-value
    p_value = compute_p_value(flips)
    p_values.append(p_value)

    # Add another flip if we haven't reached the limit
    if len(flips) < total_n:
        new_flip = np.random.binomial(
            n=1,
            p=0.5
        )
        flips = np.append(flips, new_flip)


# ----------------------------
# Plot
# ----------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    n_values,
    p_values,
    marker="o",
    linestyle="-",
    color="blue",
    label="P-value"
)

plt.axhline(
    y=0.05,
    color="red",
    linestyle="--",
    label="Significance Level (α = 0.05)"
)

plt.title("P-Value vs Number of Coin Flips")
plt.xlabel("Number of Coin Flips")
plt.ylabel("P-Value")
plt.grid(True)
plt.legend()

plt.show()