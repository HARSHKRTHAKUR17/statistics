import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from matplotlib.animation import FuncAnimation


# Generate 100 random coin flips
# 1 = heads, 0 = tails
coin_flips = np.random.choice([1, 0], size=100)


# Initial prior: Beta(1, 1)
alpha_prior = 1
beta_prior = 1


# Set up the figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

# Adjust space between subplots
plt.subplots_adjust(hspace=0.3)


# --------------------------------------------------
# First plot: Beta posterior distribution
# --------------------------------------------------

theta_range = np.linspace(0, 1, 1000)

(line1,) = ax1.plot([], [], lw=2)

(mean_line,) = ax1.plot(
    [],
    [],
    lw=2,
    linestyle="dashed",
    color="orange",
)

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 5)

ax1.set_title("Beta Distribution Evolution")
ax1.set_xlabel("Theta (probability of heads)")
ax1.set_ylabel("Density")


# --------------------------------------------------
# Second plot: Posterior mean evolution
# --------------------------------------------------

(line2,) = ax2.plot(
    [],
    [],
    lw=2,
    color="orange",
)

ax2.set_xlim(0, len(coin_flips))
ax2.set_ylim(0, 1)

ax2.set_title("Evolution of Posterior Mean")
ax2.set_xlabel("Flip Number")
ax2.set_ylabel("Posterior Mean")


# --------------------------------------------------
# Store posterior means and standard deviations
# --------------------------------------------------

posterior_means = [
    alpha_prior / (alpha_prior + beta_prior)
]

posterior_stds = [
    np.sqrt(
        (alpha_prior * beta_prior)
        / (
            (alpha_prior + beta_prior) ** 2
            * (alpha_prior + beta_prior + 1)
        )
    )
]


# --------------------------------------------------
# Initialize animation
# --------------------------------------------------

def init():
    density = beta.pdf(
        theta_range,
        alpha_prior,
        beta_prior,
    )

    line1.set_data(theta_range, density)

    mean_line.set_data([], [])
    line2.set_data([], [])

    return line1, mean_line, line2


# --------------------------------------------------
# Update animation
# --------------------------------------------------

def update(frame):
    global alpha_prior
    global beta_prior

    if frame > 0:
        flip = coin_flips[frame - 1]

        # Bayesian update
        alpha_prior += flip
        beta_prior += 1 - flip

        # Posterior mean
        posterior_mean = alpha_prior / (
            alpha_prior + beta_prior
        )

        # Posterior standard deviation
        posterior_std = np.sqrt(
            (alpha_prior * beta_prior)
            / (
                (alpha_prior + beta_prior) ** 2
                * (alpha_prior + beta_prior + 1)
            )
        )

        posterior_means.append(posterior_mean)
        posterior_stds.append(posterior_std)

    else:
        flip = None

        posterior_mean = alpha_prior / (
            alpha_prior + beta_prior
        )

        posterior_std = np.sqrt(
            (alpha_prior * beta_prior)
            / (
                (alpha_prior + beta_prior) ** 2
                * (alpha_prior + beta_prior + 1)
            )
        )


    # --------------------------------------------------
    # Update Beta distribution
    # --------------------------------------------------

    density = beta.pdf(
        theta_range,
        alpha_prior,
        beta_prior,
    )

    line1.set_data(
        theta_range,
        density,
    )

    # Dynamic y-axis
    ax1.set_ylim(
        0,
        1.1 * np.max(density),
    )


    # --------------------------------------------------
    # Update posterior mean evolution
    # --------------------------------------------------

    line2.set_data(
        np.arange(len(posterior_means)),
        posterior_means,
    )


    # --------------------------------------------------
    # Remove previous shaded regions
    # --------------------------------------------------

    for collection in list(ax1.collections):
        collection.remove()

    for collection in list(ax2.collections):
        collection.remove()


    # --------------------------------------------------
    # Shade ±1 standard deviation on Beta distribution
    # --------------------------------------------------

    ax1.fill_between(
        theta_range,
        density,
        where=(
            (theta_range >= posterior_mean - posterior_std)
            & (theta_range <= posterior_mean + posterior_std)
        ),
        color="orange",
        alpha=0.4,
    )


    # --------------------------------------------------
    # Shade ±1 standard deviation around posterior mean
    # --------------------------------------------------

    mean_array = np.array(posterior_means)
    std_array = np.array(posterior_stds)

    ax2.fill_between(
        np.arange(len(posterior_means)),
        mean_array - std_array,
        mean_array + std_array,
        color="orange",
        alpha=0.4,
    )


    # --------------------------------------------------
    # Update posterior mean vertical line
    # --------------------------------------------------

    mean_line.set_data(
        [posterior_mean, posterior_mean],
        [0, np.max(density)],
    )


    # --------------------------------------------------
    # Update title
    # --------------------------------------------------

    if frame == 0:
        title = "Prior"

    elif flip == 1:
        title = "Heads"

    else:
        title = "Tails"

    ax1.set_title(
        f"After Flip {frame}: {title}"
    )

    return line1, mean_line, line2


# --------------------------------------------------
# Create animation
# --------------------------------------------------

ani = FuncAnimation(
    fig,
    update,
    frames=len(coin_flips) + 1,
    init_func=init,
    interval=200,
    blit=False,
    repeat=False,
)


# --------------------------------------------------
# Show animation in VS Code
# --------------------------------------------------

plt.show()