import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib.cm as cm
from scipy.optimize import curve_fit

# Define sine-cosine function with exponential decay for fitting
def sincos_decay_func(t, A, B, omega, gamma, offset):
    """(A * sin(omega * t) + B * cos(omega * t)) * exp(-gamma * t) + offset"""
    return (A * np.sin(omega * t) + B * np.cos(omega * t)) * np.exp(-gamma * t) + offset


# Create output folder for plots
if not os.path.exists('./fig2b-fft-sinfit-plots'):
    os.makedirs('./fig2b-fft-sinfit-plots')

qubits = [20]
noises = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.5, 2.0]
amps = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.5, 2.0]
gs = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

# Create color maps for amp and g parameters (for future use if needed)
amp_colors = cm.viridis(np.linspace(0, 1, len(amps)))
g_colors = cm.plasma(np.linspace(0, 1, len(gs)))


for n in qubits:
    # Load fit results
    results_filename = f'sincosfit_results_qubits-{n}.csv'
    results_path = os.path.join('./fig2b-sincosfit-results', results_filename)
    
    if os.path.exists(results_path):
        fit_results_df = pd.read_csv(results_path)
        print(f"Loaded fit results from: {results_filename}")
    else:
        print(f"Warning: Fit results file not found: {results_filename}")
        fit_results_df = None
    
    # Create a figure for each g parameter showing noise vs amp table
    for g in gs:
        # Create subplot grid: rows for noise, columns for amp
        fig, axes = plt.subplots(len(noises), len(amps), figsize=(20, 15), 
                                sharex=True, 
                                # sharey=True
                                )
        fig.suptitle(rf'FFT of  $\langle Z(t) \rangle$ g={g}, L={n}', fontsize=16)

        # If only one row or column, ensure axes is 2D
        if len(noises) == 1:
            axes = axes.reshape(1, -1)
        if len(amps) == 1:
            axes = axes.reshape(-1, 1)
        if len(noises) == 1 and len(amps) == 1:
            axes = np.array([[axes]])
        
        for i, noise in enumerate(noises):
            for j, amp in enumerate(amps):
                input_file_path = f'./fig2b-data/qubits-{n}/noise-{noise}/amp-{amp}/g-{g}/data.csv'
                try:
                    data = pd.read_csv(input_file_path)
                    
                    # Get original data
                    expval = data['expval'].values
                    time = data['time'].values
                    
                    # Subtract mean to remove DC component
                    # expval_centered = expval - np.mean(expval)
                    expval_centered = expval
                    
                    # Calculate sampling frequency and frequency array
                    dt = time[1] - time[0]  # assuming uniform time spacing
                    freq = np.fft.rfftfreq(len(expval), dt)
                    
                    # Compute FFT of original data
                    fft_expval = np.fft.rfft(expval_centered)
                    fft_magnitude_orig = np.abs(fft_expval) / len(expval)
                    
                    # Plot original data FFT
                    # axes[i, j].plot(freq, fft_magnitude_orig, 'bo-', markersize=1.5, alpha=0.7, label='Original Data')
                    
                    # Add vertical lines at frequencies 1/m where m is integer from 2 to 10
                    # Color palette from sincosfit file
                    palette_colors = ['#332288', '#117733', '#882255', '#44AA99', '#88CCEE', '#DDCC77', '#CC6677', '#AA4499']
                    m_values = range(2, 11)
                    for idx, m in enumerate(m_values):
                        freq_val = 1.0 / m
                        if m == 2:  # This gives f=0.5, use first color
                            color = "#000000"
                            axes[i, j].axvline(x=freq_val, color=color, linestyle='--', 
                                             alpha=0.9, linewidth=1.5, label=rf'$f=\frac{{1}}{{{m}}}$')
                        else:  # All other lines use the same color
                            color = palette_colors[1]  # Use second color for all others
                            axes[i, j].axvline(x=freq_val, color=color, linestyle=':', 
                                             alpha=0.8, linewidth=1.5)
                    
                    # Get fitted parameters and compute fitted data if available
                    if fit_results_df is not None:
                        fit_row = fit_results_df[
                            (fit_results_df['qubits'] == n) & 
                            (fit_results_df['g'] == g) & 
                            (fit_results_df['noise'] == noise) & 
                            (fit_results_df['amp'] == amp)
                        ]
                        
                        if not fit_row.empty and fit_row.iloc[0]['fit_success']:
                            # Get fitted parameters
                            A_fit = fit_row.iloc[0]['A_fitted']
                            B_fit = fit_row.iloc[0]['B_fitted']
                            omega_fit = fit_row.iloc[0]['omega_fitted']
                            gamma_fit = fit_row.iloc[0]['gamma_fitted']
                            offset_fit = fit_row.iloc[0]['offset_fitted']
                            
                            # Generate high-resolution fitted data for better FFT
                            t_fit = np.linspace(time[0], time[-1], len(time) * 10)
                            y_fit = sincos_decay_func(t_fit, A_fit, B_fit, omega_fit, gamma_fit, offset_fit)
                            # y_fit_centered = y_fit - np.mean(y_fit)
                            y_fit_centered = y_fit
                            
                            # Compute FFT of fitted data
                            dt_fit = t_fit[1] - t_fit[0]
                            freq_fit = np.fft.rfftfreq(len(y_fit), dt_fit)
                            fft_fit = np.fft.rfft(y_fit_centered)
                            fft_magnitude_fit = np.abs(fft_fit) / len(y_fit)
                            
                            # Plot fitted data FFT
                            axes[i, j].plot(freq_fit, fft_magnitude_fit, color='#E72142', linewidth=1.5, alpha=0.8, label='Fitted Data')
                            
                            # Add vertical line at fitted frequency
                            fitted_freq = omega_fit / (2 * np.pi)
                            axes[i, j].axvline(x=fitted_freq, color='#332288', linestyle='-', alpha=0.7, linewidth=2, label=f'Fitted f={fitted_freq:.3f}')
                            

                    axes[i, j].grid(True, alpha=0.3)
                    axes[i, j].legend(fontsize=4, loc='upper right')
                    
                    # Add x-axis labels on top row
                    if i == 0:
                        axes[i, j].set_xlabel(rf'$A$={amp}', fontsize=8)
                        axes[i, j].xaxis.set_label_position('top')
                    
                    # Add y-axis labels on left column
                    if j == 0:
                        axes[i, j].set_ylabel(rf'$\delta$={noise}', fontsize=8, rotation=0, labelpad=20)
                    
                    # Set tick parameters for better readability
                    axes[i, j].tick_params(labelsize=6)
                    
                except FileNotFoundError:
                    axes[i, j].text(0.5, 0.5, 'No Data', ha='center', va='center', 
                                   transform=axes[i, j].transAxes, fontsize=10)
                    # axes[i, j].set_title(f'noise={noise}, amp={amp}', fontsize=8)
                    if i == 0:
                        axes[i, j].set_xlabel(rf'$A$={amp}', fontsize=8)
                        axes[i, j].xaxis.set_label_position('top')
                    
                    # Add y-axis labels on left column
                    if j == 0:
                        axes[i, j].set_ylabel(rf'$\delta$={noise}', fontsize=8, rotation=0, labelpad=20)
                    continue
        
        # Add single axis labels for the entire figure
        fig.supxlabel('f', fontsize=12)
        fig.supylabel(r'$|\mathcal{F}\{\langle Z(t) \rangle\}|$', fontsize=12)
        plt.xlim(0, 1.0)
        plt.tight_layout()
        
        # Save the plot with descriptive filename
        plot_filename = f'fig2b_fft_comparison_qubits-{n}_g-{g:.2f}.png'
        plot_path = os.path.join('./fig2b-fft-sinfit-plots', plot_filename)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {plot_filename}")
        
        plt.close()  # Close the figure to save memory
        
    print(f"Completed FFT comparison plots for {n} qubits")
