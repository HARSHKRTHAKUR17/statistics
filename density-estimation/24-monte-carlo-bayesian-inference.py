import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import binom, uniform
from matplotlib.animation import FuncAnimation


# ----------------------------------------------------
# True probability of heads
# ----------------------------------------------------

theta_true = 0.5

# Number of flips per update
batch_size = 10
total_flips = 250

# Generate coin flips
np.random.seed(42)
coin_flips = np.random.binomial(
    1,
    theta_true,
    total_flips,
)

# Number of Monte Carlo samples
num_samples = 5000

# Prior: Uniform(0,1)
prior_samples = np.random.uniform(
    0,
    1,
    num_samples,
)

# ----------------------------------------------------
# Create figure
# ----------------------------------------------------

fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(8, 8),
)

plt.subplots_adjust(hspace=0.3)

# ----------------------------------------------------
# Posterior density plot
# ----------------------------------------------------

theta_range = np.linspace(0, 1, 1000)

line1, = ax1.plot([], [], lw=2)

mean_line, = ax1.plot(
    [],
    [],
    lw=2,
    linestyle="dashed",
    color="orange",
)

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 10)

ax1.set_title("Monte Carlo Posterior Sampling Evolution")
ax1.set_xlabel("Theta (probability of heads)")
ax1.set_ylabel("Density")

# ----------------------------------------------------
# Posterior mean plot
# ----------------------------------------------------

line2, = ax2.plot([], [], lw=2, color="orange")

ax2.set_xlim(0, total_flips // batch_size)
ax2.set_ylim(0, 1)

ax2.set_title("Evolution of Posterior Mean")
ax2.set_xlabel("Batch Number")
ax2.set_ylabel("Posterior Mean")

# ----------------------------------------------------
# Shaded regions
# ----------------------------------------------------

std_shade1 = ax1.fill_between(
    [],
    [],
    [],
    color="orange",
    alpha=0.2,
)

std_shade2 = ax2.fill_between(
    [],
    [],
    [],
    color="orange",
    alpha=0.2,
)

posterior_means = []
posterior_stds = []
batch_numbers = []


# ----------------------------------------------------
# Initialize animation
# ----------------------------------------------------

def init():
    line1.set_data([], [])
    mean_line.set_data([], [])
    line2.set_data([], [])

    return line1, mean_line, line2


# ----------------------------------------------------
# Update animation
# ----------------------------------------------------

def update(batch_number):

    global prior_samples
    global std_shade1
    global std_shade2

    # Current batch
    start = batch_number * batch_size
    end = start + batch_size

    batch_flips = coin_flips[start:end]

    heads = np.sum(batch_flips)
    tails = batch_size - heads

    # Binomial likelihood
    likelihoods = binom.pmf(
        heads,
        batch_size,
        prior_samples,
    )

    # Normalize weights
    weights = likelihoods / np.sum(likelihoods)

    # Importance resampling
    posterior_samples = np.random.choice(
        prior_samples,
        size=num_samples,
        p=weights,
    )

    # Posterior becomes prior
    prior_samples = posterior_samples

    posterior_mean = np.mean(posterior_samples)
    posterior_std = np.std(posterior_samples)

    posterior_means.append(posterior_mean)
    posterior_stds.append(posterior_std)
    batch_numbers.append(batch_number)

    # Histogram approximation
    density, _ = np.histogram(
        posterior_samples,
        bins=theta_range,
        density=True,
    )

    line1.set_data(
        theta_range[:-1],
        density,
    )

    ax1.set_ylim(
        0,
        1.1 * np.max(density),
    )

    line2.set_data(
        batch_numbers,
        posterior_means,
    )

    # Remove previous shaded regions
    for coll in ax1.collections:
        coll.remove()

    for coll in ax2.collections:
        coll.remove()

    # ±1 standard deviation on posterior
    std_shade1 = ax1.fill_between(
        theta_range[:-1],
        density,
        where=(
            (theta_range[:-1] >= posterior_mean - posterior_std)
            &
            (theta_range[:-1] <= posterior_mean + posterior_std)
        ),
        color="orange",
        alpha=0.4,
    )

    # ±1 standard deviation on mean plot
    std_shade2 = ax2.fill_between(
        batch_numbers,
        np.array(posterior_means) - np.array(posterior_stds),
        np.array(posterior_means) + np.array(posterior_stds),
        color="orange",
        alpha=0.4,
    )

    # Posterior mean
    mean_line.set_data(
        [posterior_mean, posterior_mean],
        [0, np.max(density)],
    )

    ax1.set_title(
        f"Batch {batch_number + 1}: Posterior after {end} flips"
    )

    return line1, mean_line, line2


# ----------------------------------------------------
# Animation
# ----------------------------------------------------

ani = FuncAnimation(
    fig,
    update,
    frames=total_flips // batch_size,
    init_func=init,
    blit=True,
    repeat=False,
)

# ----------------------------------------------------
# Jupyter display
# ----------------------------------------------------

plt.show()