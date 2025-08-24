import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Load the data from the CSV file
data = pd.read_csv('autocorr_data_L20_fakebrisbane/autocorr_data_vacuum_g0.97_L20_inst1_randomphi1_delta0.0_amplitude1.0_noise0.05_usenoise1_fakebrisbane.csv')
# Extract the time and autocorrelation values
time = data['time']
av_autocorr = data['av_autocorr']
av_autocorr_echo = data['av_autocorr_echo']

# Plot the autocorrelation values
plt.figure(figsize=(10, 6))
plt.plot(time, av_autocorr, label='Average Autocorrelation', marker='o', linestyle='-')
plt.plot(time, av_autocorr_echo, label='Average Echo Autocorrelation', marker='x', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation vs Time')
plt.legend()
plt.grid()
plt.tight_layout()
# Save the plot as a PNG file
plt.savefig('autocorrelation_plot-fakebrisbane.png')
# Show the plot
plt.show()