import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib.cm as cm
from scipy.optimize import curve_fit

# Create output folder for plots
if not os.path.exists('./fig2b-sincosfit-plots'):
    os.makedirs('./fig2b-sincosfit-plots')

# Create output folder for fit results
if not os.path.exists('./fig2b-sincosfit-results'):
    os.makedirs('./fig2b-sincosfit-results')

# Define sine-cosine function with exponential decay for fitting
def sincos_decay_func(t, A, B, omega, gamma, offset):
    """(A * sin(omega * t) + B * cos(omega * t)) * exp(-gamma * t) + offset"""
    return (A * np.sin(omega * t) + B * np.cos(omega * t)) * np.exp(-gamma * t) + offset

qubits = [20]
noises = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.5, 2.0]
amps = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.5, 2.0]
gs = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
# gs = [1.0]

# Create color maps for amp and g parameters (for future use if needed)
amp_colors = cm.viridis(np.linspace(0, 1, len(amps)))
g_colors = cm.plasma(np.linspace(0, 1, len(gs)))


for n in qubits:
    # Initialize list to store fit results
    fit_results = []

    # Create a figure for each g parameter showing noise vs amp table
    for g in gs:
        # Create subplot grid: rows for noise, columns for amp
        fig, axes = plt.subplots(len(noises), len(amps), figsize=(5.7*3, 4.3*3),
                                sharex=True, sharey=True)
        fig.suptitle(rf"Fitted $\langle Z(t) \rangle$ g={g}, L={n}", fontsize=12)

        # If only one row or column, ensure axes is 2D
        if len(noises) == 1:
            axes = axes.reshape(1, -1)
        if len(amps) == 1:
            axes = axes.reshape(-1, 1)
        if len(noises) == 1 and len(amps) == 1:
            axes = np.array([[axes]])
        # color palette
        # #332288-#117733-#882255-#44AA99-#88CCEE-#DDCC77-#CC6677-#AA4499
        # print("#332288"-"#117733"-"#882255"-"#44AA99"-"#88CCEE"-"#DDCC77"-"#CC6677"-"#AA4499")
        for i, noise in enumerate(noises):
            for j, amp in enumerate(amps):
                input_file_path = f'./fig2b-data/qubits-{n}/noise-{noise}/amp-{amp}/g-{g}/data.csv'
                try:
                    data = pd.read_csv(input_file_path)
                    t_data = data['time'].values
                    y_data = data['expval'].values

                    # Plot original data - emphasize the oscillations
                    axes[i, j].plot(t_data, y_data, 'o-', color='#332288', markersize=2, alpha=1, label=r'$\langle Z(t) \rangle$', markerfacecolor='#332288', markeredgecolor='#332288', markeredgewidth=0.8, linewidth=1.5, zorder=3)

                    # Attempt to fit sin+cos with decay function
                    try:
                        # Initial parameter guesses
                        A_guess = np.clip((np.max(y_data) - np.min(y_data)) / 2, -1, 1)
                        B_guess = 0.0
                        offset_guess = np.mean(y_data)
                        gamma_guess = 0.1
                        # Estimate frequency from data
                        if len(t_data) > 10:
                            fft_freqs = np.fft.fftfreq(len(t_data), d=np.mean(np.diff(t_data)))
                            fft_vals = np.abs(np.fft.fft(y_data - np.mean(y_data)))
                            idx = np.argmax(fft_vals[1:len(fft_vals)//2]) + 1
                            omega_guess = 2 * np.pi * np.abs(fft_freqs[idx])
                            if omega_guess < 1e-3:
                                omega_guess = 1.0
                        else:
                            omega_guess = 1.0

                        # Fit the sin+cos*exp(-gamma*t) function with amplitude constraints
                        popt, pcov = curve_fit(
                            sincos_decay_func, t_data, y_data,
                            p0=[A_guess, B_guess, omega_guess, gamma_guess, offset_guess],
                            bounds=([-1, -1, 0, 0, -np.inf], [1, 1, np.inf, np.inf, np.inf]),
                            maxfev=5000
                        )

                        # Create high-resolution time array for smooth plotting
                        t_min, t_max = np.min(t_data), np.max(t_data)
                        t_fit = np.linspace(t_min, t_max, len(t_data) * 10)
                        y_fit = sincos_decay_func(t_fit, *popt)

                        # Plot fitted curve - visible but secondary
                        axes[i, j].plot(t_fit, y_fit, '-', color='#E72142', linewidth=1.0, alpha=0.65, label='Fit', zorder=2)

                        # Add fit parameters as text
                        A, B, omega, gamma, offset = popt
                        freq = omega / (2 * np.pi)
                        fit_text = rf'$C={A:.3f}$ $D={B:.3f}$ $f={freq:.3f}$ $\gamma={gamma:.3f}$'
                        axes[i, j].text(0.02, 0.02, fit_text, transform=axes[i, j].transAxes,
                                       fontsize=5, verticalalignment='bottom',
                                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                        # Store fit results
                        fit_results.append({
                            'qubits': n,
                            'g': g,
                            'noise': noise,
                            'amp': amp,
                            'A_fitted': A,
                            'B_fitted': B,
                            'omega_fitted': omega,
                            'frequency_fitted': freq,
                            'gamma_fitted': gamma,
                            'offset_fitted': offset,
                            'fit_success': True
                        })

                    except Exception as fit_error:
                        print(f"Fitting failed for noise={noise}, amp={amp}, g={g}: {fit_error}")
                        # Store failed fit result
                        fit_results.append({
                            'qubits': n,
                            'g': g,
                            'noise': noise,
                            'amp': amp,
                            'A_fitted': np.nan,
                            'B_fitted': np.nan,
                            'omega_fitted': np.nan,
                            'frequency_fitted': np.nan,
                            'gamma_fitted': np.nan,
                            'offset_fitted': np.nan,
                            'fit_success': False
                        })

                    axes[i, j].grid(True, alpha=0.3)
                    axes[i, j].set_ylim(-1.05, 1.05)
                    axes[i, j].legend(fontsize=5, loc='upper right')

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
                    axes[i, j].set_ylim(-1.05, 1.05)
                    axes[i, j].text(0.5, 0.5, 'No Data', transform=axes[i, j].transAxes,
                                   ha='center', va='center', fontsize=8, alpha=0.5)
                    # Store no data result
                    fit_results.append({
                        'qubits': n,
                        'g': g,
                        'noise': noise,
                        'amp': amp,
                        'A_fitted': np.nan,
                        'B_fitted': np.nan,
                        'omega_fitted': np.nan,
                        'frequency_fitted': np.nan,
                        'offset_fitted': np.nan,
                        'fit_success': False
                    })
                    # Add x-axis labels on top row even for missing data
                    if i == 0:
                        axes[i, j].set_xlabel(rf'$A$={amp}', fontsize=8)
                        axes[i, j].xaxis.set_label_position('top')

                    # Add y-axis labels on left column even for missing data
                    if j == 0:
                        axes[i, j].set_ylabel(rf'$\delta$={noise}', fontsize=8, rotation=0, labelpad=20)
                    continue

        # Add single axis labels for the entire figure
        fig.supxlabel('t (FT)', fontsize=12)
        fig.supylabel(r'$\langle Z(t) \rangle$', fontsize=12)

        plt.tight_layout()

        # Save the plot with descriptive filename
        plot_filename = f'fig2b_sincosfit_qubits-{n}_g-{g:.2f}.png'
        plot_path = os.path.join('./fig2b-sincosfit-plots', plot_filename)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {plot_filename}")

        plt.close()  # Close the figure to save memory

    # Save fit results to CSV file
    if fit_results:
        results_df = pd.DataFrame(fit_results)
        results_filename = f'sincosfit_results_qubits-{n}.csv'
        results_path = os.path.join('./fig2b-sincosfit-results', results_filename)
        results_df.to_csv(results_path, index=False)
        print(f"Saved fit results: {results_filename}")

    print(f"Completed sin+cos-fitted plots for {n} qubits")
