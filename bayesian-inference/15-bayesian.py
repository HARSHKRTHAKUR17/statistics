# Simple example where we have two hypotheses

import numpy as np

# Prior probabilities for fair (H1) and biased (H2) coin
P_H1 = 0.5  # Prior for H1: Coin is fair
P_H2 = 0.5  # Prior for H2: Coin is biased (P(Heads) = 0.7)


# Likelihood of the data under each hypothesis
def likelihood(data, hypothesis):
    if hypothesis == "H1":  # Fair coin
        return 0.5

    elif hypothesis == "H2":  # Biased coin
        return 0.7 if data == 1 else 0.3


# Update beliefs using Bayes' Theorem
def bayesian_update(prior_H1, prior_H2, data):
    # Likelihoods
    likelihood_H1 = likelihood(data, "H1")
    likelihood_H2 = likelihood(data, "H2")

    # Total evidence P(D)
    P_data = likelihood_H1 * prior_H1 + likelihood_H2 * prior_H2

    # Posterior probabilities
    posterior_H1 = (likelihood_H1 * prior_H1) / P_data
    posterior_H2 = (likelihood_H2 * prior_H2) / P_data

    return posterior_H1, posterior_H2


# Generate 20 random coin flips
coin_flips = np.random.choice([1, 0], size=20)

P_H1_posterior = P_H1
P_H2_posterior = P_H2

# Perform Bayesian updates after each flip
for flip in coin_flips:
    P_H1_posterior, P_H2_posterior = bayesian_update(
        P_H1_posterior,
        P_H2_posterior,
        flip
    )

    print(
        f"After observing {flip}: "
        f"P(H1 | Data) = {P_H1_posterior:.4f}, "
        f"P(H2 | Data) = {P_H2_posterior:.4f}"
    )