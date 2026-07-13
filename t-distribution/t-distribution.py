import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, norm

# Range of x values for plotting
x = np.linspace(-5, 5, 1000)

# Degrees of freedom to demonstrate convergence
degrees_of_freedom = [1, 2, 5, 10, 30, 100]

# Modern way to get the colormap
cmap = plt.get_cmap("viridis_r")

# Set the background color to white
plt.figure(figsize=(10, 6), facecolor="white")
ax = plt.gca()
ax.set_facecolor("white")

# Plot the standard normal distribution
plt.plot(
    x,
    norm.pdf(x),
    "k--",
    lw=2,
    label="Standard Normal Distribution"
)

# Plot the t-distributions
num_colors = len(degrees_of_freedom)

for i, df in enumerate(degrees_of_freedom):
    plt.plot(
        x,
        t.pdf(x, df),
        lw=2,
        color=cmap(i / (num_colors - 1)),
        label=f"t-distribution (df={df})"
    )

# Labels, title, and legend
plt.title(
    "Convergence of t-Distribution to Normal Distribution",
    fontsize=14
)

plt.xlabel("x", fontsize=12)
plt.ylabel("PDF", fontsize=12)

plt.legend(
    facecolor="white",
    framealpha=1,
    edgecolor="black",
    labelcolor="black"
)

# Adjust tick color
ax.tick_params(colors="black")

# Grid
plt.grid(True, color="gray", alpha=0.5)

plt.show()