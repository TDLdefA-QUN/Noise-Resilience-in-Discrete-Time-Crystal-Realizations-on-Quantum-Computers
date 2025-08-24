import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def load_autocorr_data(folder_path):
    data = {}
    
    # Find all autocorr CSV files
    autocorr_files = glob.glob(os.path.join(folder_path, "autocorr_data_*.csv"))
    
    for file in autocorr_files:
        filename = os.path.basename(file)
        # Extract noise level from filename
        if "noise0.005" in filename:
            noise_level = "0.005"
        elif "noise0.05" in filename:
            noise_level = "0.05"
        else:
            continue
            
        df = pd.read_csv(file)
        data[noise_level] = df
    
    return data

def load_x_noisy_data():
    """Load L20 polarization data for comparison"""
    l4_data = {}
    
    # Define the L20 files with different noise levels
    l4_files = {
        "0.05": r"D:\METU\MSc\TEZ\Time crsytal\analyze expvalzt\autocorr_polarization\autocorr_data_L20_polarization\autocorr_data_vacuum_g0.97_L20_inst30_randomphi1_delta0.0_amplitude1.0_noise0.05_usenoise1_polx.csv",
        "0.005": r"D:\METU\MSc\TEZ\Time crsytal\analyze expvalzt\autocorr_polarization\autocorr_data_L20_polarization\autocorr_data_vacuum_g0.97_L20_inst30_randomphi1_delta0.0_amplitude1.0_noise0.005_usenoise1_polx.csv"
    }
    
    for noise_level, l4_file in l4_files.items():
        if os.path.exists(l4_file):
            df = pd.read_csv(l4_file)
            l4_data[noise_level] = df
            print(f"Loaded L20 polarization data for noise level {noise_level}")
        else:
            print(f"L20 polarization file not found: {l4_file}")
    
    return l4_data

def plot_autocorr_comparison(xy_data, x_data, save_dir="xy-cycle-plots"):
    """Create comparison plots for autocorr data with separate subplots for different noise levels"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Determine how many subplots we need based on available noise levels
    noise_levels = list(set(xy_data.keys()) | set(x_data.keys()))
    noise_levels.sort()
    
    if len(noise_levels) == 1:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, len(noise_levels), figsize=(8 * len(noise_levels), 8), sharey=True)
        if len(noise_levels) == 1:
            axes = [axes]
    
    fig.suptitle('Autocorrelation Comparison: XY Cycle vs X Polarization', fontsize=16)
    
    for i, noise_level in enumerate(noise_levels):
        ax = axes[i] if len(noise_levels) > 1 else axes[0]
        
        # Plot XY cycle data if available
        if noise_level in xy_data:
            df_xy = xy_data[noise_level]
            ax.plot(df_xy['time'], df_xy['av_autocorr'], 'o-', 
                   label=f'X-Y (Noise {noise_level})', 
                   alpha=0.8, markersize=5, color='darkgreen', linewidth=2)
            
            pos_xy_filter = df_xy['av_autocorr'] > 0
            neg_xy_filter = df_xy['av_autocorr'] < 0

            pos_xy = df_xy['av_autocorr'][pos_xy_filter]
            neg_xy = df_xy['av_autocorr'][neg_xy_filter]
            pos_xy_time = df_xy['time'][pos_xy_filter]
            neg_xy_time = df_xy['time'][neg_xy_filter]


            
            
            ax.plot(df_xy['time'], df_xy['av_autocorr_echo'], 's--', 
                   label=f'X-Y Echo (Noise {noise_level})', 
                   alpha=0.8, markersize=5, color='orange', linewidth=2)
        
        # Plot Only X Polarization data if available
        if noise_level in x_data:
            df_x = x_data[noise_level]
            ax.plot(df_x['time'], df_x['av_autocorr'], '^-',
                   label=f'X Polarization (Noise {noise_level})',
                   alpha=0.8, markersize=5, color='darkblue', linewidth=2)
            ax.plot(df_x['time'], df_x['av_autocorr_echo'], 'd--',
                   label=f'X Polarization Echo (Noise {noise_level})',  
                   alpha=0.8, markersize=5, color='red', linewidth=2)
        
        # Add vertical lines every 5 seconds
        max_time = 0
        if noise_level in xy_data:
            max_time = max(max_time, xy_data[noise_level]['time'].max())
        if noise_level in x_data:
            max_time = max(max_time, x_data[noise_level]['time'].max())

        for t in range(0, int(max_time) + 1, 5):
            if t > 0:  # Don't draw line at t=0
                ax.axvline(x=t, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Average Autocorrelation')
        ax.set_title(f'Noise Level {noise_level}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Fix axis ticks
        ax.set_xticks(range(0, int(max_time) + 1, 2))  # Major ticks every 2 seconds
        ax.set_xticks(np.arange(0, max_time + 1, 1), minor=True)  # Minor ticks every 1 second
        ax.tick_params(axis='x', which='minor', length=3)
        ax.tick_params(axis='x', which='major', length=6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'xy_vs_l4_autocorr_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()

def main():
    folder_path = r"d:\METU\MSc\TEZ\Time crsytal\analyze expvalzt\autocorr_polarization\autocorr_data_L20_polarization_xy_cycle"
    
    xy_autocorr_data = load_autocorr_data(folder_path)
    
    x_autocorr_data = load_x_noisy_data()
    
    plot_autocorr_comparison(xy_autocorr_data, x_autocorr_data)
    


if __name__ == "__main__":
    main()