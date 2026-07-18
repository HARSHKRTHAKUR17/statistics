import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity
from matplotlib.animation import FuncAnimation, PillowWriter



# ---------------------------------------------------------
# Generate 100 random coin flips
# (1 = Heads, 0 = Tails)
# ---------------------------------------------------------

coin_flips = np.random.choice([1, 0], size=100)

# ---------------------------------------------------------
# Initial empirical prior (Uniform)
# ---------------------------------------------------------

prior_samples = np.random.uniform(0, 1, 1000)

# ---------------------------------------------------------
# Create figure
# ---------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
plt.subplots_adjust(hspace=0.3)

# ---------------------------------------------------------
# Density plot
# ---------------------------------------------------------

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
ax1.set_title("Empirical Distribution Evolution")
ax1.set_xlabel("Theta (probability of heads)")
ax1.set_ylabel("Density")

# ---------------------------------------------------------
# Posterior mean plot
# ---------------------------------------------------------

line2, = ax2.plot([], [], lw=2, color="orange")

ax2.set_xlim(0, len(coin_flips))
ax2.set_ylim(0, 1)

ax2.set_title("Evolution of Posterior Mean")
ax2.set_xlabel("Flip Number")
ax2.set_ylabel("Posterior Mean")

# ---------------------------------------------------------
# Initial shaded regions
# ---------------------------------------------------------

std_shade1 = ax1.fill_between([], [], [], color="orange", alpha=0.2)
std_shade2 = ax2.fill_between([], [], [], color="orange", alpha=0.2)

posterior_means = []
posterior_stds = []

# ---------------------------------------------------------
# Initialize KDE
# ---------------------------------------------------------

kde_prior = KernelDensity(
    kernel="gaussian",
    bandwidth=0.05,
).fit(prior_samples[:, np.newaxis])

# ---------------------------------------------------------
# Empirical likelihood
# ---------------------------------------------------------

def empirical_likelihood(data, samples):
    heads = np.sum(data == 1)
    tails = np.sum(data == 0)

    return (samples ** heads) * ((1 - samples) ** tails)

# ---------------------------------------------------------
# Initialize animation
# ---------------------------------------------------------

def init():
    line1.set_data([], [])
    mean_line.set_data([], [])
    line2.set_data([], [])

    return line1, mean_line, line2

# ---------------------------------------------------------
# Animation update
# ---------------------------------------------------------

def update(frame):

    global prior_samples, std_shade1, std_shade2

    flip = coin_flips[frame]

    # Likelihood
    likelihoods = empirical_likelihood(
        np.array([flip]),
        prior_samples,
    )

    # Importance weights
    posterior_weights = likelihoods / np.sum(likelihoods)

    # Resample posterior
    posterior_samples = np.random.choice(
        prior_samples,
        size=1000,
        p=posterior_weights,
    )

    # Posterior becomes new prior
    prior_samples = posterior_samples

    # KDE
    kde_prior = KernelDensity(
        kernel="gaussian",
        bandwidth=0.05,
    ).fit(posterior_samples[:, np.newaxis])

    log_density = kde_prior.score_samples(
        theta_range[:, np.newaxis]
    )

    density = np.exp(log_density)

    line1.set_data(theta_range, density)

    posterior_mean = np.mean(posterior_samples)
    posterior_std = np.std(posterior_samples)

    posterior_means.append(posterior_mean)
    posterior_stds.append(posterior_std)

    line2.set_data(
        np.arange(len(posterior_means)),
        posterior_means,
    )

    # Remove previous confidence bands
    for coll in ax1.collections:
        coll.remove()

    for coll in ax2.collections:
        coll.remove()

    # ±1 std on KDE
    std_shade1 = ax1.fill_between(
        theta_range,
        density,
        where=(
            (theta_range >= posterior_mean - posterior_std)
            &
            (theta_range <= posterior_mean + posterior_std)
        ),
        color="orange",
        alpha=0.4,
    )

    # ±1 std on mean evolution
    std_shade2 = ax2.fill_between(
        np.arange(len(posterior_means)),
        np.array(posterior_means) - np.array(posterior_stds),
        np.array(posterior_means) + np.array(posterior_stds),
        color="orange",
        alpha=0.4,
    )

    # Mean line
    mean_line.set_data(
        [posterior_mean, posterior_mean],
        [0, 10],
    )

    # Title
    ax1.set_title(
        f'After Flip {frame + 1}: '
        f'{"Heads" if flip == 1 else "Tails"}'
    )

    return line1, mean_line, line2

# ---------------------------------------------------------
# Animation
# ---------------------------------------------------------

ani = FuncAnimation(
    fig,
    update,
    frames=len(coin_flips),
    init_func=init,
    blit=True,
    repeat=False,
)

# ---------------------------------------------------------
# Jupyter Notebook
# ---------------------------------------------------------

plt.show()