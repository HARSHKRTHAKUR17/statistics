# Alpha particles emitted by Americium 241
# Example from Rice, taken from Berkson 1966
# 10220 Geiger clicks (emissions) were measured, specifically time between clicks
# Data is binned into 10-second intervals
# Clicks per 10-second interval should be Poisson (time between is exponential)

import numpy as np
import matplotlib.pyplot as plt

# Data
ClicksPerTenSeconds = np.arange(18)   # Number of counts per 10-second interval
ObservedCounts = [
    1, 6, 11, 28, 56, 105, 126, 146,
    164, 161, 123, 101, 74, 53, 23,
    15, 9, 5
]

# Plotting the observed data
if __name__ == "__main__":
    plt.figure(figsize=(10, 6))
    plt.plot(
        ClicksPerTenSeconds,
        ObservedCounts,
        marker='o',
        linestyle='--',
        color='blue'
    )

    # Adding labels and title
    plt.title('Observed Alpha Particle Emissions per 10-Second Interval', fontsize=14)
    plt.xlabel('Number of Clicks per 10-Second Interval', fontsize=12)
    plt.ylabel('Observed Count', fontsize=12)

    plt.grid(True)

    # Set x-axis to display integer tick marks only
    plt.xticks(ticks=np.arange(0, max(ClicksPerTenSeconds) + 1, 1))

    # Show the plot

    plt.show()