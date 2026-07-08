from scipy.stats import poisson
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

N=10000
lambda_value = 10
DataPopulation = np.random.poisson(lam=lambda_value, size=N)

mean_population = np.mean(DataPopulation)
# Generate a histogram of DataPopulation
plt.figure(figsize=(10, 6))
counts, bins, _ = plt.hist(
    DataPopulation,
    bins=19,
    color='blue',
    edgecolor='black',
    alpha=0.7,
    density=True
)

# Plot the Poisson distribution that best fits the sample mean
x = np.arange(0, bins[-1] + 1)
pmf_values = poisson.pmf(x, mu=mean_population)
plt.plot(
    x,
    pmf_values,
    'k-',
    marker='o',
    label=f'Poisson PMF (mean = {mean_population:.2f})'
)

# Label the axes
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.title('Histogram of DataPopulation with Fitted Poisson Distribution')

# Add a legend
plt.legend()

# Display the plot
plt.show()