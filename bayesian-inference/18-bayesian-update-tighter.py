import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from matplotlib.animation import FuncAnimation


# Generate 100 random coin flips
# 1 for heads, 0 for tails
coin_flips = np.random.choice([1, 0], size=100)


# Initial prior: Beta(1, 1) distribution (uniform prior)
alpha_prior = 1
beta_prior = 1


# Set up the figure with two subplots:
# one for the density and one for the mean
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))


# Adjust the space between the subplots
plt.subplots_adjust(hspace=0.3)


# ---------------------------------------------------------
# Set up the first plot for Beta distribution
# ---------------------------------------------------------

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

# Set initial y-limit; we'll update this dynamically later
ax1.set_ylim(0, 5)

ax1.set_title("Beta Distribution Evolution")
ax1.set_xlabel("Theta (probability of heads)")
ax1.set_ylabel("Density")


# ---------------------------------------------------------
# Set up the second plot for mean evolution
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Initialize shaded regions for both plots
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Store the evolution of mean and std after each update
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Function to initialize the plots
# ---------------------------------------------------------

def init():
    # Plot the initial prior distribution Beta(1, 1)
    density = beta.pdf(
        theta_range,
        alpha_prior,
        beta_prior,
    )

    line1.set_data(theta_range, density)

    mean_line.set_data([], [])
    line2.set_data([], [])

    return line1, mean_line, line2


# ---------------------------------------------------------
# Function to update plots after every 10 coin flips
# ---------------------------------------------------------

def update(frame):
    global alpha_prior, beta_prior
    global std_shade1, std_shade2

    # If frame == 0, we are showing the prior,
    # so don't update
    if frame > 0:

        # Accumulate 10 flips at a time
        flips = coin_flips[
            (frame - 1) * 10 : frame * 10
        ]

        # Number of heads in the last 10 flips
        heads = np.sum(flips)

        # Number of tails in the last 10 flips
        tails = len(flips) - heads

        # Update the Beta distribution parameters
        alpha_prior += heads
        beta_prior += tails


    # -----------------------------------------------------
    # Calculate posterior mean and standard deviation
    # -----------------------------------------------------

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


    # Only append if it's an update
    if frame > 0:
        posterior_means.append(posterior_mean)
        posterior_stds.append(posterior_std)


    # -----------------------------------------------------
    # Update Beta distribution
    # -----------------------------------------------------

    density = beta.pdf(
        theta_range,
        alpha_prior,
        beta_prior,
    )

    line1.set_data(
        theta_range,
        density,
    )


    # Dynamically update the y-axis limit
    ax1.set_ylim(
        0,
        1.1 * np.max(density),
    )


    # -----------------------------------------------------
    # Update posterior mean evolution
    # -----------------------------------------------------

    line2.set_data(
        np.arange(len(posterior_means)) * 10,
        posterior_means,
    )


    # -----------------------------------------------------
    # Remove previous shaded regions
    # -----------------------------------------------------

    for coll in list(ax1.collections):
        coll.remove()

    for coll in list(ax2.collections):
        coll.remove()


    # -----------------------------------------------------
    # Update shaded region for Beta distribution
    # ±1 standard deviation
    # -----------------------------------------------------

    std_shade1 = ax1.fill_between(
        theta_range,
        beta.pdf(
            theta_range,
            alpha_prior,
            beta_prior,
        ),
        where=(
            (theta_range >= posterior_mean - posterior_std)
            & (theta_range <= posterior_mean + posterior_std)
        ),
        color="orange",
        alpha=0.4,
    )


    # -----------------------------------------------------
    # Update shaded region for posterior mean
    # ±1 standard deviation
    # -----------------------------------------------------

    std_shade2 = ax2.fill_between(
        np.arange(len(posterior_means)) * 10,
        np.array(posterior_means)
        - np.array(posterior_stds),
        np.array(posterior_means)
        + np.array(posterior_stds),
        color="orange",
        alpha=0.4,
    )


    # -----------------------------------------------------
    # Update vertical dashed line for posterior mean
    # -----------------------------------------------------

    mean_line.set_data(
        [posterior_mean, posterior_mean],
        [0, np.max(density)],
    )


    # -----------------------------------------------------
    # Update title
    # -----------------------------------------------------

    if frame == 0:
        ax1.set_title(
            "Prior Distribution (Beta(1,1))"
        )

    else:
        ax1.set_title(
            f"After Flip {frame * 10}: "
            f"Heads = {heads}, Tails = {tails}"
        )


    return line1, mean_line, line2


# ---------------------------------------------------------
# Create animation
# ---------------------------------------------------------

# Initial prior + one frame for every 10 flips
frames = len(coin_flips) // 10 + 1

ani = FuncAnimation(
    fig,
    update,
    frames=frames,
    init_func=init,
    interval=700,
    blit=False,
    repeat=False,
)


# ---------------------------------------------------------
# Display animation in VS Code
# ---------------------------------------------------------

plt.show()