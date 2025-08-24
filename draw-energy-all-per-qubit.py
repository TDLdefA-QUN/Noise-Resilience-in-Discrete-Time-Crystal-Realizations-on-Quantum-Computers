import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks

def find_envelope(signal, window_size=5):
    """Find the upper and lower envelope of a signal using rolling max/min with interpolation for smoothness"""
    from scipy.interpolate import interp1d
    from scipy.signal import find_peaks
    
    signal = np.array(signal)
    time_indices = np.arange(len(signal))
    
    # Method 1: Find peaks and valleys for smoother envelope
    # Find peaks (maxima) with some minimum distance
    peaks_max, _ = find_peaks(signal, distance=max(1, window_size//2))
    peaks_min, _ = find_peaks(-signal, distance=max(1, window_size//2))
    
    # Always include endpoints for better interpolation
    if 0 not in peaks_max:
        peaks_max = np.concatenate([[0], peaks_max])
    if len(signal)-1 not in peaks_max:
        peaks_max = np.concatenate([peaks_max, [len(signal)-1]])
        
    if 0 not in peaks_min:
        peaks_min = np.concatenate([[0], peaks_min])
    if len(signal)-1 not in peaks_min:
        peaks_min = np.concatenate([peaks_min, [len(signal)-1]])
    
    # Sort the indices
    peaks_max = np.sort(peaks_max)
    peaks_min = np.sort(peaks_min)
    
    # Create smooth envelopes using interpolation
    if len(peaks_max) >= 2:
        # Cubic spline interpolation for upper envelope
        f_upper = interp1d(peaks_max, signal[peaks_max], kind='cubic', 
                          bounds_error=False, fill_value='extrapolate')
        upper_env = f_upper(time_indices)
    else:
        # Fallback to constant if not enough points
        upper_env = np.full_like(signal, np.max(signal))
    
    if len(peaks_min) >= 2:
        # Cubic spline interpolation for lower envelope
        f_lower = interp1d(peaks_min, signal[peaks_min], kind='cubic', 
                          bounds_error=False, fill_value='extrapolate')
        lower_env = f_lower(time_indices)
    else:
        # Fallback to constant if not enough points
        lower_env = np.full_like(signal, np.min(signal))
    
    # Ensure upper envelope is always >= signal and lower envelope is always <= signal
    upper_env = np.maximum(upper_env, signal)
    lower_env = np.minimum(lower_env, signal)
    
    # Apply light smoothing to remove any remaining sharp edges
    from scipy.ndimage import gaussian_filter1d
    sigma = max(0.5, window_size/4)  # Adaptive smoothing based on window size
    upper_env = gaussian_filter1d(upper_env, sigma=sigma)
    lower_env = gaussian_filter1d(lower_env, sigma=sigma)
    
    # Final check to ensure envelopes bound the signal properly
    upper_env = np.maximum(upper_env, signal)
    lower_env = np.minimum(lower_env, signal)
    
    return upper_env, lower_env

nprobs = [0, 0.001, 0.01, 0.1, "iqm", "ibm"]  # Added IQM and IBM to the list
ts = np.arange(0, 20, 1)

# Read CSV file
csv_filename = "energy-data_L20/energy_data_vacuum_g0.97_L20_inst1_randomphi1_delta0.0_amplitude1.0_noise0.05_usenoise1.csv"
df = pd.read_csv(csv_filename)

csv_filename = "energy-data_L20-iqm/energy_data_vacuum_g0.97_L20_inst1_randomphi1_delta0.0_amplitude1.0_noise0.05_usenoise1.csv"
df2 = pd.read_csv(csv_filename)

csv_filename = "energy-data_L127-ibm/energy_data_vacuum_g0.97_L127_inst1_randomphi1_delta0.0_amplitude1.0_noise0.05_usenoise1.csv"
df3 = pd.read_csv(csv_filename)
# Extract energy data for each noise probability
energies = []
for nprob in nprobs:
    if nprob == "iqm":
        # Handle IQM data from second CSV file
        if 'energy_p_iqm' in df2.columns:
            energies.append(df2['energy_p_iqm'].values/20)
        else:
            print(f"Warning: Column energy_p_iqm not found in IQM CSV")
            energies.append(np.zeros(len(ts)))
    elif nprob == "ibm":
        # Handle IBM data from third CSV file
        if 'energy_p_ibm' in df3.columns:
            energies.append(df3['energy_p_ibm'].values/127)
        else:
            print(f"Warning: Column energy_p_ibm not found in IBM CSV")
            energies.append(np.zeros(len(ts)))
    else:
        # Handle numeric noise probabilities from first CSV file
        column_name = f'energy_p_{nprob}'
        if column_name in df.columns:
            energies.append(df[column_name].values/20)
        else:
            print(f"Warning: Column {column_name} not found in CSV")
            energies.append(np.zeros(len(ts)))
# Set parameters for plotting
state = "vacuum"
g = 0.97
L = 20
inst = 1
tf = 20 
plt.figure(figsize=(14, 10))

# Define contrasting colors for better visibility (added one more for IQM)
colors = ['#000000',  '#1f77b4', '#ff7f0e', '#2ca02c', "#803ac2", "#d62728"]  # Black, Blue, Orange, Green, Purple, Red
fit_colors = ['#333333',  '#0d4f8c', '#cc5500', '#1a6b1a', "#634091", "#811717"]  # Darker versions for fits

# Plot original data and envelope fits
for i, e in enumerate(energies):
    color = colors[i % len(colors)]
    fit_color = fit_colors[i % len(fit_colors)]
    plt.plot(ts, e, 'o-', color=color, label=f'$p = {nprobs[i]}$ (data)', alpha=0.8, markersize=6, linewidth=2)
    
    # Find and plot envelopes using simple rolling window approach
    try:
        upper_env, lower_env = find_envelope(e, window_size=3)
        
        # Plot envelope as filled area
        plt.fill_between(ts, lower_env, upper_env, alpha=0.2, color=color)
        
        print(f"Noise prob {nprobs[i]}: Envelope computed successfully")
        
    except Exception as err:
        print(f"Envelope computation failed for noise prob {nprobs[i]}: {str(err)}")
        # Fallback to simple data plot
        plt.plot(ts, e, '-', color=fit_color, label=f'$p = {nprobs[i]}$ (envelope failed)', alpha=0.8)

# Find and print minimum energies for all data
print("\n" + "="*60)
print("MINIMUM ENERGY ANALYSIS")
print("="*60)
for i, e in enumerate(energies):
    min_energy = np.min(e)
    # Use correct number of qubits: 127 for IBM, 20 for others
    num_qubits = 127 if nprobs[i] == "ibm" else L
    min_energy_per_qubit = min_energy / num_qubits
    min_time_idx = np.argmin(e)
    min_time = ts[min_time_idx]
    print(f"Noise prob {nprobs[i]:>6}: Min Energy = {min_energy:.6f}, Per Qubit = {min_energy_per_qubit:.6f} (L={num_qubits}) at t = {min_time}")

# Find overall minimum across all data
all_mins = [np.min(e) for e in energies]
all_mins_per_qubit = []
for i, min_val in enumerate(all_mins):
    num_qubits = 127 if nprobs[i] == "ibm" else L
    all_mins_per_qubit.append(min_val / num_qubits)

overall_min = np.min(all_mins)
overall_min_per_qubit = np.min(all_mins_per_qubit)
overall_min_idx = np.argmin(all_mins)
overall_min_per_qubit_idx = np.argmin(all_mins_per_qubit)
overall_min_nprob = nprobs[overall_min_idx]
overall_min_per_qubit_nprob = nprobs[overall_min_per_qubit_idx]

print(f"\nOVERALL MINIMUM (absolute): {overall_min:.6f} (noise prob {overall_min_nprob})")
print(f"OVERALL MINIMUM (per qubit): {overall_min_per_qubit:.6f} (noise prob {overall_min_per_qubit_nprob})")
print("="*60 + "\n")

plt.ylim(-4,0.3)
plt.xlabel('Time $t$')
plt.ylabel('Energy $E$')
plt.legend(loc='upper left', framealpha=0.9, fontsize=8, ncol=6)
plt.title(f'Energy for {state} state ($g={g}$, $L={L}$) with Envelopes')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
folder_name = "energy-data_L20-ibm-vs-iqm-vs-simulation-per-qubit"
import os
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

plot_filename = f"energy_plot_{state}_g{g}_L{L}_inst{inst}_tf{tf}.png"
plot_path = f"{folder_name}/{plot_filename}"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {plot_path}")

plt.show()