from scipy.stats import poisson
import numpy as np
import matplotlib.pyplot as plt

# importing vars without changing file names in python
import importlib.util
from pathlib import Path

module_path = Path(__file__).parent / "07-plot-poiss-data.py"

spec = importlib.util.spec_from_file_location("plot_data", module_path)
plot_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot_data)


# Calculate the sample mean
sample_mean = np.average(plot_data.ClicksPerTenSeconds, weights=plot_data.ObservedCounts)

# Fit a Poisson distribution using the sample mean as lambda (rate parameter)
poisson_dist = (
    poisson.pmf(plot_data.ClicksPerTenSeconds, mu=sample_mean)
    * np.sum(plot_data.ObservedCounts)
)

# Plotting the observed data
plt.figure(figsize=(10, 6))

plt.plot(
    plot_data.ClicksPerTenSeconds,
    plot_data.ObservedCounts,
    marker='o',
    linestyle='--',
    color='blue',
    label='Observed Data'
)

# Plot the Poisson distribution
plt.plot(
    plot_data.ClicksPerTenSeconds,
    poisson_dist,
    marker='x',
    linestyle='--',
    color='red',
    label=f'Poisson Fit (λ = {sample_mean:.2f})'
)

# Adding labels and title
plt.title('Observed Alpha Particle Emissions per 10-Second Interval', fontsize=14)
plt.xlabel('Number of Clicks per 10-Second Interval', fontsize=12)
plt.ylabel('Observed Count', fontsize=12)

plt.grid(True)
plt.legend()

# Set x-axis to display integer tick marks only
plt.xticks(ticks=np.arange(0, max(plot_data.ClicksPerTenSeconds) + 1, 1))

plt.show()