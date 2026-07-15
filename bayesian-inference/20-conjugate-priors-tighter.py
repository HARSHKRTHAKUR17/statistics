import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from matplotlib.animation import FuncAnimation


# Generate 100 random coin flips (1 for heads, 0 for tails)
coin_flips = np.random.choice([1, 0], size=100)


# Initial prior: Beta(1, 1) distribution (uniform prior)
alpha_prior = 1
beta_prior = 1


# Set up the figure with two subplots:
# one for the density and one for the mean
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))


# Adjust the space between the subplots
plt.subplots_adjust(hspace=0.3)


# Set up the first plot for Beta distribution
theta_range = np.linspace(0, 1, 1000)

(line1,) = ax1.plot([], [], lw=2)

(mean_line,) = ax1.plot(
    [],
    [],
    lw=2,
    linestyle="dashed",
    color="orange",
)  # Dashed line for mean

ax1.set_xlim(0, 1)

# Set initial y-limit; we'll update this dynamically later
ax1.set_ylim(0, 5)

ax1.set_title("Beta Distribution Evolution")
ax1.set_xlabel("Theta (probability of heads)")
ax1.set_ylabel("Density")


# Set up the second plot for mean evolution
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


# Initialize shaded regions for both plots
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


# To store the evolution of the mean and std after each flip
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


# Function to initialize the plots
def init():
    # Plot the initial prior distribution (Beta(1, 1))
    density = beta.pdf(
        theta_range,
        alpha_prior,
        beta_prior,
    )

    line1.set_data(theta_range, density)

    mean_line.set_data([], [])
    line2.set_data([], [])

    return line1, mean_line, line2


# Function to update the plots after each coin flip
def update(frame):
    global alpha_prior, beta_prior
    global std_shade1, std_shade2

    # If frame == 0, we are showing the prior, so don't update
    if frame > 0:
        flip = coin_flips[frame - 1]

        # Update the Beta distribution parameters
        alpha_prior += flip
        beta_prior += 1 - flip

    else:
        flip = None


    # Calculate the posterior mean and standard deviation
    # for Beta(alpha, beta)
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


    # Only append if it's a flip
    if frame > 0:
        posterior_means.append(posterior_mean)
        posterior_stds.append(posterior_std)


    # Update the line for the Beta distribution (density plot)
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
    # based on the max value of the density
    ax1.set_ylim(
        0,
        1.1 * np.max(density),
    )


    # Update the line for the evolution of the mean
    line2.set_data(
        np.arange(len(posterior_means)),
        posterior_means,
    )


    # Remove the previous shaded region
    # properly clearing it
    for coll in list(ax1.collections):
        coll.remove()

    for coll in list(ax2.collections):
        coll.remove()


    # Update the shaded region for the Beta distribution plot
    # ±1 std deviation
    std_shade1 = ax1.fill_between(
        theta_range,
        beta.pdf(
            theta_range,
            alpha_prior,
            beta_prior,
        ),
        where=(
            (
                theta_range
                >= posterior_mean - posterior_std
            )
            & (
                theta_range
                <= posterior_mean + posterior_std
            )
        ),
        color="orange",
        alpha=0.4,
    )


    # Update the shaded region for the mean plot
    # ±1 std deviation
    std_shade2 = ax2.fill_between(
        np.arange(len(posterior_means)),
        np.array(posterior_means)
        - np.array(posterior_stds),
        np.array(posterior_means)
        + np.array(posterior_stds),
        color="orange",
        alpha=0.4,
    )


    # Update the vertical dashed line for the mean
    # in the Beta plot
    mean_line.set_data(
        [posterior_mean, posterior_mean],
        [0, np.max(density)],
    )


    # Update titles
    ax1.set_title(
        f"After Flip {frame}: "
        f"{'Prior' if frame == 0 else 'Heads' if flip == 1 else 'Tails'}"
    )


    return line1, mean_line, line2


# Create the animation,
# including the initial prior (frame 0)
ani = FuncAnimation(
    fig,
    update,
    frames=len(coin_flips) + 1,
    init_func=init,
    interval=200,
    blit=False,
    repeat=False,
)


# Display the animation in VS Code
plt.show()