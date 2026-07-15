import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta


# Function to compute the likelihood of the data given theta
def likelihood(theta, data):
    # Binomial likelihood for coin flips (Bernoulli trials)
    heads = np.sum(data == "H")
    tails = np.sum(data == "T")

    return theta**heads * (1 - theta) ** tails


# Function to compute the posterior distribution using Bayesian updating
def bayesian_update(prior_alpha, prior_beta, data):
    # Calculate the number of heads and tails in the data
    heads = np.sum(data == "H")
    tails = np.sum(data == "T")

    # Update the prior (Beta distribution) parameters
    posterior_alpha = prior_alpha + heads
    posterior_beta = prior_beta + tails

    return posterior_alpha, posterior_beta


# Prior: Uniform distribution Beta(1, 1)
prior_alpha = 1
prior_beta = 1


# Simulate a sequence of coin flips
# H for heads, T for tails
coin_flips = np.array([
    "H", "T", "H", "H",
    "T", "H", "T", "T"
])


# Store prior values for plotting
alphas = [prior_alpha]
betas = [prior_beta]


# Perform Bayesian updates after each flip
for i, flip in enumerate(coin_flips):
    # Update the posterior with the new data (one flip)
    prior_alpha, prior_beta = bayesian_update(
        prior_alpha,
        prior_beta,
        np.array([flip]),
    )

    # Store the updated posterior parameters
    alphas.append(prior_alpha)
    betas.append(prior_beta)

    # Print the updated posterior parameters after each flip
    print(
        f"After observing {i + 1} flip(s) ({flip}): "
        f"Posterior alpha = {prior_alpha}, "
        f"beta = {prior_beta}"
    )


# Plot the evolution of the posterior distributions
theta_range = np.linspace(0, 1, 100)


plt.figure(figsize=(10, 6))


# Modern Matplotlib colormap API
cmap = plt.get_cmap("viridis_r")

# Generate evenly spaced colors
colors = cmap(
    np.linspace(0, 1, len(alphas))
)


# Plot each posterior distribution
for i in range(len(alphas)):
    posterior_pdf = beta.pdf(
        theta_range,
        alphas[i],
        betas[i],
    )

    plt.plot(
        theta_range,
        posterior_pdf,
        color=colors[i],
        label=f"After {i} flips",
    )


# Customize the plot
plt.title("Posterior Distributions After Each Coin Flip")

plt.xlabel("Theta (probability of heads)")
plt.ylabel("Density")

plt.legend()

plt.show()