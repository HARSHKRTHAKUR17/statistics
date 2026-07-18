import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity


# Function to compute the empirical likelihood of the data given samples
def likelihood(samples, data):
    # Binomial likelihood for coin flips (Bernoulli trials)
    heads = np.sum(data == "H")
    tails = np.sum(data == "T")

    return samples**heads * (1 - samples) ** tails


# Function to update the posterior distribution using KDE
def bayesian_update_kde(prior_samples, data):
    # Compute the empirical likelihood of the data
    likelihoods = likelihood(prior_samples, data)

    # Weight the prior samples by the likelihood
    posterior_weights = likelihoods / np.sum(likelihoods)

    # Resample from the prior based on the likelihoods
    # to generate posterior samples
    posterior_samples = np.random.choice(
        prior_samples,
        size=1000,
        p=posterior_weights,
    )

    # Use Kernel Density Estimation to estimate the posterior distribution
    kde = KernelDensity(
        kernel="gaussian",
        bandwidth=0.05,
    ).fit(posterior_samples[:, np.newaxis])

    return posterior_samples, kde


# -----------------------------------------------------
# Prior: Uniform distribution (between 0 and 1)
# -----------------------------------------------------

prior_samples = np.random.uniform(0, 1, 1000)


# Simulate a sequence of coin flips
# H for heads, T for tails
coin_flips = np.array([
    "H",
    "T",
    "H",
    "H",
    "T",
    "H",
    "T",
    "T",
])


# Store prior samples for plotting
samples_list = [prior_samples]


# -----------------------------------------------------
# Perform Bayesian updates after each flip
# -----------------------------------------------------

for i, flip in enumerate(coin_flips):

    # Update the posterior using the KDE method
    prior_samples, kde = bayesian_update_kde(
        prior_samples,
        np.array([flip]),
    )

    # Store the updated posterior samples
    samples_list.append(prior_samples)

    # Print posterior mean
    print(
        f"After observing {i+1} flip(s) ({flip}): "
        f"Posterior mean = {np.mean(prior_samples):.4f}"
    )


# -----------------------------------------------------
# Plot the evolution of the posterior distributions
# -----------------------------------------------------

theta_range = np.linspace(0, 1, 1000)

plt.figure(figsize=(10, 6))


# Use a colormap
cmap = plt.get_cmap("viridis_r")
colors = cmap(np.linspace(0, 1, len(samples_list)))


for i, samples in enumerate(samples_list):

    # Estimate the PDF using KDE
    kde = KernelDensity(
        kernel="gaussian",
        bandwidth=0.05,
    ).fit(samples[:, np.newaxis])

    log_density = kde.score_samples(
        theta_range[:, np.newaxis]
    )

    density = np.exp(log_density)

    # Plot the posterior distribution
    plt.plot(
        theta_range,
        density,
        color=colors[i],
        label=f"After {i} flips",
    )


# -----------------------------------------------------
# Customize the plot
# -----------------------------------------------------

plt.title("Posterior Distributions After Each Coin Flip (KDE)")
plt.xlabel("Theta (probability of heads)")
plt.ylabel("Density")
plt.legend()

plt.show()