import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

def complex_func(x, a, b, c, d, e, f, g):
    """Complex function: a*(x+b)^c + d*log(e*x+f) + g"""
    return a * np.power(x + b, c) + d * np.log(e * x + f) + g

def linear_func(x, m, b):
    """Linear function: mx + b"""
    return m * x + b

def find_envelope_points(ts, e, window_size=3):
    """Find local maxima and minima for envelope"""
    # Find peaks (maxima)
    peaks_max, _ = find_peaks(e, distance=window_size)
    # Find valleys (minima) by inverting the signal
    peaks_min, _ = find_peaks(-e, distance=window_size)
    
    # Add endpoints if they're not already included
    if 0 not in peaks_max and 0 not in peaks_min:
        if e[0] > e[1]:
            peaks_max = np.concatenate([[0], peaks_max])
        else:
            peaks_min = np.concatenate([[0], peaks_min])
    
    if len(e)-1 not in peaks_max and len(e)-1 not in peaks_min:
        if e[-1] > e[-2]:
            peaks_max = np.concatenate([peaks_max, [len(e)-1]])
        else:
            peaks_min = np.concatenate([peaks_min, [len(e)-1]])
    
    return peaks_max, peaks_min

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
            energies.append(df2['energy_p_iqm'].values)
        else:
            print(f"Warning: Column energy_p_iqm not found in IQM CSV")
            energies.append(np.zeros(len(ts)))
    elif nprob == "ibm":
        # Handle IBM data from third CSV file
        if 'energy_p_ibm' in df3.columns:
            energies.append(df3['energy_p_ibm'].values)
        else:
            print(f"Warning: Column energy_p_ibm not found in IBM CSV")
            energies.append(np.zeros(len(ts)))
    else:
        # Handle numeric noise probabilities from first CSV file
        column_name = f'energy_p_{nprob}'
        if column_name in df.columns:
            energies.append(df[column_name].values)
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
    
    # Find envelope points (maxima and minima)
    try:
        peaks_max, peaks_min = find_envelope_points(ts, e, window_size=2)
        
        # Extract envelope points
        ts_max = ts[peaks_max]
        e_max = e[peaks_max]
        ts_min = ts[peaks_min]
        e_min = e[peaks_min]
        
        print(f"Noise prob {nprobs[i]}: Found {len(ts_max)} max points, {len(ts_min)} min points")
        
        # Fit upper envelope (maxima) - include t=0 with special handling
        if len(ts_max) >= 2:  # Reduced from > 2 to >= 2
            fit_ts_max = ts_max
            fit_e_max = e_max
            
            try:
                # For complex function, include t=0 in fitting since all y values are same at t=0
                # This provides valuable constraint information
                if 0 in fit_ts_max:
                    # Include t=0 but handle log function carefully
                    # For t=0, we have: y = a*(0+b)^c + d*log(e*0+f) + g = a*b^c + d*log(f) + g
                    # So we need b > 0 (if c can be negative) and f > 0 for log(f) to be defined
                    
                    # Use all points including t=0
                    fit_ts_max_for_fit = fit_ts_max
                    fit_e_max_for_fit = fit_e_max
                    
                    # Complex function fit for upper envelope (including t=0)
                    # Parameters: [a, b, c, d, e, f, g]
                    initial_guess_max = [1.0, 1.0, -0.5, 1.0, 1.0, 1.0, np.mean(fit_e_max)]
                    bounds_max = ([-1000, 0.001, -5, -1000, 0.001, 0.001, -1000],  # More relaxed bounds
                                 [1000, 1000, 5, 1000, 1000, 1000, 1000])
                    popt_max, _ = curve_fit(complex_func, fit_ts_max_for_fit, fit_e_max_for_fit, 
                                          p0=initial_guess_max, bounds=bounds_max, maxfev=20000)
                    a_max, b_max, c_max, d_max, e_max, f_max, g_max = popt_max
                    
                    # Generate smooth upper envelope starting from 0
                    t_smooth = np.linspace(0, max(ts), 100)
                    e_upper = complex_func(t_smooth, a_max, b_max, c_max, d_max, e_max, f_max, g_max)
                else:
                    # Standard complex function fit without t=0
                    initial_guess_max = [1.0, 1.0, -0.5, 1.0, 1.0, 0.1, np.mean(fit_e_max)]
                    bounds_max = ([-1000, 0.001, -5, -1000, 0.001, 0.001, -1000],  # More relaxed bounds
                                 [1000, 1000, 5, 1000, 1000, 1000, 1000])
                    popt_max, _ = curve_fit(complex_func, fit_ts_max, fit_e_max, 
                                          p0=initial_guess_max, bounds=bounds_max, maxfev=20000)
                    a_max, b_max, c_max, d_max, e_max, f_max, g_max = popt_max
                    
                    # Generate smooth upper envelope
                    t_smooth = np.linspace(max(0.1, min(fit_ts_max)), max(ts), 100)
                    e_upper = complex_func(t_smooth, a_max, b_max, c_max, d_max, e_max, f_max, g_max)
                
                plt.plot(t_smooth, e_upper, '--', color=fit_color, alpha=0.9, linewidth=3,
                        # label=f'$p = {nprobs[i]}$ upper fit'
                        )
                
                print(f"  Upper envelope: a={a_max:.6f}, b={b_max:.6f}, c={c_max:.6f}, d={d_max:.6f}, e={e_max:.6f}, f={f_max:.6f}, g={g_max:.6f}")
                print(f"  Upper fit equation: {a_max:.2f}(t+{b_max:.2f})^{c_max:.3f} + {d_max:.2f}ln({e_max:.2f}t + {f_max:.2f}) + {g_max:.2f}")
                
            except Exception as e_err:
                print(f"  Upper envelope fitting failed: {e_err}")
                # Plot the envelope points even if fitting fails
                plt.plot(ts_max, e_max, 's', color=fit_color, markersize=8, alpha=0.8,
                        label=f'$p = {nprobs[i]}$ max points')
                e_upper = None
        else:
            print(f"  Not enough max points ({len(ts_max)}) for upper envelope fitting")
            # Plot the max points anyway
            if len(ts_max) > 0:
                plt.plot(ts_max, e_max, '^', color=fit_color, markersize=10, alpha=0.8,
                        label=f'$p = {nprobs[i]}$ max points only')
            e_upper = None
        
        # Fit lower envelope (minima) - include t=0 since all y values are same at t=0
        if len(ts_min) >= 2:  # Reduced from > 2 to >= 2
            # Include t=0 in fitting since it provides valuable constraint
            fit_ts_min = ts_min
            fit_e_min = e_min
            
            try:
                # Complex function fit for lower envelope (including t=0)
                # At t=0: y = a*b^c + d*log(f) + g, so we need b > 0 and f > 0
                initial_guess_min = [1.0, 1.0, -0.5, 1.0, 1.0, 1.0, np.mean(fit_e_min)]
                bounds_min = ([-1000, 0.001, -5, -1000, 0.001, 0.001, -1000],  # More relaxed bounds
                             [1000, 1000, 5, 1000, 1000, 1000, 1000])
                popt_min, _ = curve_fit(complex_func, fit_ts_min, fit_e_min, 
                                      p0=initial_guess_min, bounds=bounds_min, maxfev=20000)
                a_min, b_min, c_min, d_min, e_min, f_min, g_min = popt_min
                
                # Generate smooth lower envelope including t=0
                t_smooth_lower = np.linspace(0, max(ts), 100)
                e_lower = complex_func(t_smooth_lower, a_min, b_min, c_min, d_min, e_min, f_min, g_min)
                
                plt.plot(t_smooth_lower, e_lower, '--', color=fit_color, alpha=0.9, linewidth=3,
                        # label=f'$p = {nprobs[i]}$ lower fit'
                        )
                
                print(f"  Lower envelope: a={a_min:.6f}, b={b_min:.6f}, c={c_min:.6f}, d={d_min:.6f}, e={e_min:.6f}, f={f_min:.6f}, g={g_min:.6f}")
                print(f"  Lower fit equation: {a_min:.2f}(t+{b_min:.2f})^{c_min:.3f} + {d_min:.2f}ln({e_min:.2f}t + {f_min:.2f}) + {g_min:.2f}")
                
            except Exception as e_err:
                print(f"  Lower envelope fitting failed: {e_err}")
                # Plot the envelope points even if fitting fails
                plt.plot(ts_min, e_min, 's', color=fit_color, markersize=8, alpha=0.8,
                        label=f'$p = {nprobs[i]}$ min points')
                e_lower = None
        else:
            print(f"  Not enough min points ({len(ts_min)}) for lower envelope fitting")
            # Plot the min points anyway
            if len(ts_min) > 0:
                plt.plot(ts_min, e_min, 'v', color=fit_color, markersize=10, alpha=0.8,
                        label=f'$p = {nprobs[i]}$ min points only')
            e_lower = None
        
        # Shade the area between upper and lower envelopes
        if e_upper is not None and e_lower is not None:
            # Use the same time array for both envelopes for proper shading
            t_shade = np.linspace(0, max(ts), 100)
            e_upper_shade = complex_func(t_shade, a_max, b_max, c_max, d_max, e_max, f_max, g_max)
            e_lower_shade = complex_func(t_shade, a_min, b_min, c_min, d_min, e_min, f_min, g_min)
            plt.fill_between(t_shade, e_lower_shade, e_upper_shade, alpha=0.4, color=color)
            
    except Exception as err:
        print(f"Envelope fitting failed for noise prob {nprobs[i]}: {str(err)}")
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

plt.xlabel('Time $t$')
plt.ylabel('Energy $E$')
plt.legend(loc='upper left', framealpha=0.9, fontsize=8, ncol=6)
plt.title(f'Energy for {state} state ($g={g}$, $L={L}$) with Envelope Fits')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
folder_name = "energy-data_L20-ibm-vs-iqm-vs-simulation"
import os
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

plot_filename = f"energy_plot_{state}_g{g}_L{L}_inst{inst}_tf{tf}.png"
plot_path = f"{folder_name}/{plot_filename}"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {plot_path}")

plt.show()