import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def load_polarization_data(folder_path):
    """Load and organize polarization autocorr data from the folder"""
    data = {}
    
    # Find all autocorr CSV files for individual polarizations
    autocorr_files = glob.glob(os.path.join(folder_path, "autocorr_data_*_pol*.csv"))
    
    for file in autocorr_files:
        filename = os.path.basename(file)
        
        # Extract noise level and polarization
        noise_level = None
        if "noise0.005" in filename:
            noise_level = "0.005"
        elif "noise0.05" in filename:
            noise_level = "0.05"
        else:
            continue
        
        # Extract polarization
        polarization = None
        if "_polx.csv" in filename:
            polarization = "x"
        elif "_poly.csv" in filename:
            polarization = "y"
        elif "_polxy.csv" in filename:
            polarization = "xy"
        elif "_polyx.csv" in filename:
            polarization = "yx"
        else:
            continue
            
        df = pd.read_csv(file)
        key = f"{polarization}_noise_{noise_level}"
        data[key] = df
    
    return data

def load_comparison_data(folder_path):
    """Load comparison data that contains all polarizations in one file"""
    data = {}
    
    # Find comparison files
    comparison_files = glob.glob(os.path.join(folder_path, "autocorr_data_comparison_*.csv"))
    
    for file in comparison_files:
        filename = os.path.basename(file)
        
        # Extract noise level
        if "noise0.005" in filename:
            noise_level = "0.005"
        elif "noise0.05" in filename:
            noise_level = "0.05"
        else:
            continue
            
        df = pd.read_csv(file)
        data[noise_level] = df
    
    return data

def plot_polarization_comparison(data, save_dir="plots"):
    """Create comparison plots for polarization data"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Polarization Autocorrelation Comparison: Different Noise Levels', fontsize=16)
    
    # Define color pairs for each polarization (light and dark variants for different noise levels)
    pol_color_pairs = {
        'x': {'0.005': 'lightblue', '0.05': 'navy'},
        'y': {'0.005': 'lightcoral', '0.05': 'darkred'},
        'xy': {'0.005': 'lightgreen', '0.05': 'darkgreen'},
        'yx': {'0.005': 'plum', '0.05': 'purple'}
    }
    
    # Define line styles for different noise levels
    noise_styles = {
        '0.005': {'linestyle': '-', 'linewidth': 2.5, 'alpha': 0.8},
        '0.05': {'linestyle': '--', 'linewidth': 3, 'alpha': 0.9}
    }
    
    # Define markers for different polarizations
    pol_markers = {
        'x': 'o',
        'y': 's',
        'xy': '^', 
        'yx': 'd'
    }
    
    # Plot normal autocorr and echo autocorr in separate subplots
    for key, df in data.items():
        polarization, _, noise_level = key.split('_')
        color = pol_color_pairs[polarization][noise_level]
        style = noise_styles[noise_level]
        marker = pol_markers[polarization]
        
        # Plot normal autocorr in first subplot
        axes[0].plot(df['time'], df['av_autocorr'], 
                     color=color, marker=marker, 
                     linestyle=style['linestyle'], linewidth=style['linewidth'],
                     label=f'{polarization.upper()} (Noise {noise_level})', 
                     alpha=style['alpha'], markersize=5, markevery=2)
        
        # Plot echo autocorr in second subplot
        axes[1].plot(df['time'], df['av_autocorr_echo'], 
                     color=color, marker=marker,
                     linestyle=style['linestyle'], linewidth=style['linewidth'],
                     label=f'{polarization.upper()} (Noise {noise_level})', 
                     alpha=style['alpha'], markersize=5, markevery=2)
    
    # Configure both subplots
    for i, (ax, title) in enumerate(zip(axes, ['Normal Autocorrelation', 'Echo Autocorrelation'])):
        # Add vertical lines every 5 seconds
        max_time = max([df['time'].max() for df in data.values()])
        for t in range(0, int(max_time) + 1, 5):
            if t > 0:  # Don't draw line at t=0
                ax.axvline(x=t, color='gray', linestyle=':', alpha=0.4, linewidth=1)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Average Autocorrelation')
        ax.set_title(title)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Fix axis ticks
        ax.set_xticks(range(0, int(max_time) + 1, 1))  # Major ticks every 1 second
        ax.set_xticks(np.arange(0, max_time + 1, 0.5), minor=True)  # Minor ticks every 0.5 seconds
        ax.tick_params(axis='x', which='minor', length=3)
        ax.tick_params(axis='x', which='major', length=6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'polarization_autocorr_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_comparison_data_analysis(comparison_data, save_dir="plots"):
    """Plot data from the comparison files that have all polarizations"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Polarization Comparison from Combined Data Files', fontsize=16)
    
    # Define polarizations and their corresponding columns
    polarizations = ['x', 'y', 'xy', 'yx']
    colors = ['blue', 'red', 'green', 'purple']
    
    # Define styles for different noise levels and data types
    noise_styles = {
        '0.005': {'normal': {'linestyle': '-', 'linewidth': 2, 'marker': 'o'}, 
                  'echo': {'linestyle': '--', 'linewidth': 2, 'marker': 's'}},
        '0.05': {'normal': {'linestyle': '-', 'linewidth': 3, 'marker': 'o'}, 
                 'echo': {'linestyle': '--', 'linewidth': 3, 'marker': 's'}}
    }
    
    for noise_level, df in comparison_data.items():
        for i, pol in enumerate(polarizations):
            ax = axes[i//2, i%2]
            
            # Plot normal and echo autocorrelation for this polarization
            normal_col = f'av_autocorr_{pol}'
            echo_col = f'av_autocorr_echo_{pol}'
            
            if normal_col in df.columns and echo_col in df.columns:
                normal_style = noise_styles[noise_level]['normal']
                echo_style = noise_styles[noise_level]['echo']
                
                ax.plot(df['time'], df[normal_col], 
                       color=colors[i], 
                       linestyle=normal_style['linestyle'],
                       linewidth=normal_style['linewidth'],
                       marker=normal_style['marker'],
                       label=f'Normal Noise {noise_level}', 
                       alpha=0.8, markersize=4, markevery=2)
                ax.plot(df['time'], df[echo_col], 
                       color=colors[i],
                       linestyle=echo_style['linestyle'],
                       linewidth=echo_style['linewidth'],
                       marker=echo_style['marker'],
                       label=f'Echo Noise {noise_level}', 
                       alpha=0.8, markersize=4, markevery=2)
            
            # Add vertical lines every 5 seconds
            max_time = df['time'].max()
            for t in range(0, int(max_time) + 1, 5):
                if t > 0:
                    ax.axvline(x=t, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
            
            ax.set_xlabel('Time')
            ax.set_ylabel('Average Autocorrelation')
            ax.set_title(f'Polarization {pol.upper()}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Fix axis ticks
            ax.set_xticks(range(0, int(max_time) + 1, 2))
            ax.set_xticks(np.arange(0, max_time + 1, 1), minor=True)
            ax.tick_params(axis='x', which='minor', length=3)
            ax.tick_params(axis='x', which='major', length=6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'polarization_combined_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_noise_effect_on_polarizations(data, save_dir="plots"):
    """Compare noise effects across different polarizations"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    fig.suptitle('Noise Effect on Different Polarizations (Echo Autocorrelation)', fontsize=16)
    
    polarizations = ['x', 'y', 'xy', 'yx']
    noise_levels = ['0.005', '0.05']
    
    # Define color pairs for each polarization (light and dark variants for different noise levels)
    pol_color_pairs = {
        'x': {'0.005': 'lightblue', '0.05': 'navy'},
        'y': {'0.005': 'lightcoral', '0.05': 'darkred'},
        'xy': {'0.005': 'lightgreen', '0.05': 'darkgreen'},
        'yx': {'0.005': 'plum', '0.05': 'purple'}
    }
    
    # Define markers for different polarizations
    pol_markers = {'x': 'o', 'y': 's', 'xy': '^', 'yx': 'd'}
    
    for i, pol in enumerate(polarizations):
        for j, noise in enumerate(noise_levels):
            key = f"{pol}_noise_{noise}"
            if key in data:
                df = data[key]
                linestyle = '-' if noise == '0.005' else '--'
                linewidth = 2.5 if noise == '0.005' else 3
                alpha = 0.8 if noise == '0.005' else 0.9
                marker = pol_markers[pol]
                color = pol_color_pairs[pol][noise]
                
                ax.plot(df['time'], df['av_autocorr_echo'], 
                       color=color, linestyle=linestyle, linewidth=linewidth,
                       marker=marker, markersize=5, markevery=2,
                       label=f'{pol.upper()} Echo (Noise {noise})', 
                       alpha=alpha)
    
    # Add vertical lines every 5 seconds
    max_time = max([df['time'].max() for df in data.values()])
    for t in range(0, int(max_time) + 1, 5):
        if t > 0:
            ax.axvline(x=t, color='gray', linestyle=':', alpha=0.4, linewidth=1)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Average Echo Autocorrelation')
    ax.set_title('Echo Autocorrelation vs Time for Different Polarizations and Noise Levels')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Fix axis ticks
    ax.set_xticks(range(0, int(max_time) + 1, 1))
    ax.set_xticks(np.arange(0, max_time + 1, 0.5), minor=True)
    ax.tick_params(axis='x', which='minor', length=3)
    ax.tick_params(axis='x', which='major', length=6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'polarization_noise_effect_echo.png'), dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to load and analyze all polarization data"""
    folder_path = r"d:\METU\MSc\TEZ\Time crsytal\analyze expvalzt\autocorr_polarization\autocorr_data_L20_polarization"
    
    print("Loading individual polarization data...")
    polarization_data = load_polarization_data(folder_path)
    print(f"Found polarization data for: {list(polarization_data.keys())}")
    
    print("Loading comparison data...")
    comparison_data = load_comparison_data(folder_path)
    print(f"Found comparison data for noise levels: {list(comparison_data.keys())}")
    
    print("Creating polarization comparison plots...")
    plot_polarization_comparison(polarization_data)
    
    print("Creating comparison data analysis...")
    plot_comparison_data_analysis(comparison_data)
    
    print("Creating noise effect analysis...")
    plot_noise_effect_on_polarizations(polarization_data)
    
    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    for key, df in polarization_data.items():
        polarization, _, noise_level = key.split('_')
        print(f"\nPolarization {polarization.upper()} (Noise {noise_level}):")
        print(f"  Initial autocorr: {df['av_autocorr'].iloc[0]:.6f}")
        print(f"  Final autocorr: {df['av_autocorr'].iloc[-1]:.6f}")
        print(f"  Initial echo autocorr: {df['av_autocorr_echo'].iloc[0]:.6f}")
        print(f"  Final echo autocorr: {df['av_autocorr_echo'].iloc[-1]:.6f}")
        print(f"  Max time: {df['time'].max()}")
        
        # Calculate decay characteristics
        abs_autocorr = np.abs(df['av_autocorr'])
        initial_val = abs_autocorr.iloc[0]
        final_val = abs_autocorr.iloc[-1]
        decay_factor = final_val / initial_val
        print(f"  Decay factor: {decay_factor:.6f}")

if __name__ == "__main__":
    main()