from scipy.stats import norm
import numpy as np
import matplotlib as plt
# Calculate the population mean and standard deviation
mean_population = np.mean(DataPopulation)
std_population = np.std(DataPopulation)

# Compute the standard deviation of the sample means (Standard Error of the Mean)
std_error = std_population / np.sqrt(n)

# Generate the normal distribution that approximates the distribution of sample means
x = np.linspace(min(SampleMeans), max(SampleMeans), 100)
normal_approximation = norm.pdf(
    x,
    loc=mean_population,
    scale=std_error
)

# Plot the histogram of the SampleMeans
plt.figure(figsize=(10, 6))
plt.hist(
    SampleMeans,
    bins=15,
    color='red',
    edgecolor='black',
    alpha=0.7,
    density=True
)

# Plot the normal distribution approximation on top of the histogram
plt.plot(
    x,
    normal_approximation,
    'b-',
    label=f'Normal Approximation\n(mean={mean_population:.2f}, std={std_error:.2f})'
)

# Label the axes and add a title
plt.xlabel('Sample Mean')
plt.ylabel('Density')
plt.title('Histogram of Sample Means with Normal Approximation')

# Add a legend
plt.legend()

# Display the plot
plt.show()