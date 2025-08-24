
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import argparse
import pandas as pd
from functools import partial
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
# from qiskit.providers.aer.noise import NoiseModel, depolarizing_error
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
)
from qiskit.quantum_info import Statevector, Operator
# from qiskit.circuit.library import RXGate, RZGate, HGate, CZGate, XGate, YGate, ZGate
from qiskit.transpiler import generate_preset_pass_manager, CouplingMap, Layout
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
from qiskit.visualization import plot_circuit_layout, plot_coupling_map
import os
from qiskit.primitives import BackendEstimatorV2
from qiskit.quantum_info import SparsePauliOp
from scipy import signal as scipy_signal

parser = argparse.ArgumentParser(description="Plot figures")
parser.add_argument("--L", type=int, default=4, help="Number of qubits")
parser.add_argument("--device_name", type=int, default=0, help="Device name")
parser.add_argument("--inst", type=int, default=1, help="Number of instances for fig3d")
parser.add_argument("--randomphi", type=int, default=1, help="Prethermal=0 or DTC=1")
parser.add_argument("--phi_delta", type=float, default=0.0, help="Phi delta parameter")
parser.add_argument("--phi_amplitude", type=float, default=1.0, help="Phi amplitude parameter")
parser.add_argument("--tf",type=int, default=100, help="end time for fig3d")
parser.add_argument("--g",type=float, default=0.97, help="g for fig3d")
parser.add_argument("--noise_prob",type=float, default=0.05, help="noise probability")
parser.add_argument("--use_noise",type=int, default=1, help="use depolarizing noise: 0=no noise, 1=apply noise")
parser.add_argument("--initial_state",type=str, default="vacuum", help="initial state")
parser.add_argument("--use_fakebackend", type=int, default=1, help="Use FakeBackend for simulation: 0=No, 1=Yes")

args = parser.parse_args()

L = args.L
g = args.g
inst = args.inst
phi_delta = args.phi_delta
phi_amplitude = args.phi_amplitude
use_noise = args.use_noise

t_start = 0
t_end = args.tf
T = t_end - t_start
ts = np.arange(t_start, t_end, 1)
noise_prob = args.noise_prob
initial_state = args.initial_state


folder_name = f"energy-data_L{L}-ham-comparison"

if not os.path.exists(folder_name):
    os.makedirs(folder_name)


disorder_folder = "."
hs_filename = os.path.join(disorder_folder, f"hs_L{L}.csv")
phis_filename = os.path.join(disorder_folder, f"phis_L{L}.csv")

hs_df = pd.read_csv(hs_filename, comment='#', header=0)
phis_df = pd.read_csv(phis_filename, comment='#', header=0)
hs = hs_df.iloc[:inst].values
phis = phis_df.iloc[:inst].values

noise_model = NoiseModel()
if args.use_fakebackend == 1:
    noise_model = NoiseModel.from_backend(FakeBrisbane())
    use_backend_noise = True  # Flag to indicate we're using backend noise
else:
    use_backend_noise = False  # We'll add custom depolarizing noise
 



def get_hamiltonian(L, g, phis, hs, instance=0, include_x_terms=True):
    ham = []
    z_str = "I" * (L)
    zz_str = "I" * (L)
    x_str = "I" * (L)
    
    # Z terms (disorder)
    for i in range(L):
        z = z_str[:i] + 'Z' + z_str[i+1:]
        ham.append((z, hs[i]))

    # ZZ terms (interaction)
    for i in range(L-1):
        zz = zz_str[:i] + 'ZZ' + zz_str[i+2:]
        ham.append((zz, phis[i]))

    # X terms (transverse field) - only include if flag is True
    if include_x_terms:
        for i in range(L):
            z = x_str[:i] + 'X' + x_str[i+1:]
            ham.append((z, g*np.pi))
    
    hamiltonian = SparsePauliOp.from_list(ham, num_qubits=L)
    return hamiltonian

def compute_z_expectation(counts: dict, num_qubits: int):
    total_shots = sum(counts.values())
    z_expectations = []

    for qubit in range(num_qubits):
        p0 = 0
        p1 = 0
        for bitstring, count in counts.items():
            # Reverse bitstring because Qiskit uses little-endian: qubit 0 is rightmost
            bit = bitstring[::-1][qubit]
            if bit == '0':
                p0 += count
            else:
                p1 += count
        expectation = (p0 - p1) / total_shots
        z_expectations.append(expectation)

    return z_expectations

def create_UF_subcircuit(L, g, phis, hs):
    subcircuit = QuantumCircuit(L)
    for i in range(L):
        subcircuit.rx(np.pi*g, i)
    for i in range(0, L-1, 2):
        subcircuit.rzz(phis[i], i, i+1)
    for i in range(1, L-1, 2):
        subcircuit.rzz(phis[i], i, i+1)
    for i in range(L):
        subcircuit.rz(hs[i], i)
    return subcircuit


def qc_qiskit(initial_state, L, g, hs, phis, t, qubit, echo=False, noise_model=None, include_x_terms=True):
    circ = QuantumCircuit(L)
    if initial_state == "neel":
        for i in range(1, L+1):
            if i % 2 == 0:
                circ.x(i)
    UF_subcircuit = create_UF_subcircuit(L, g, phis, hs)
    for _ in range(t):
        circ.append(UF_subcircuit, range(L))
    if echo:
        UF_inv = UF_subcircuit.inverse()
        for _ in range(t):
            circ.append(UF_inv, range(L))

    
    # if args.use_fakebackend == 1:
    #     backend = FakeBrisbane()
    #     print("Using FakeBrisbane backend.\n")
    # else:
        
    backend = AerSimulator(noise_model=noise_model, device="GPU",cuStateVec_enable=True)
    
    
    passmanager = generate_preset_pass_manager(
        backend=backend,
    )
    echo_str = "echo" if echo else "forward"
    backend_name = getattr(backend, 'name', str(type(backend).__name__))
    circ_tnoise = passmanager.run(circ)
    # gate_counts = circ_tnoise.count_ops()
    # gate_count_filename = f"{folder_name}/gate_counts_t{t}_{echo_str}.csv"

    # pd.DataFrame(list(gate_counts.items()), columns=["gate", "count"]).to_csv(gate_count_filename, index=False)
    
    estimator = BackendEstimatorV2(backend=backend)
    hamiltonian = get_hamiltonian(L, g, phis, hs, include_x_terms=include_x_terms)
    results = estimator.run([(circ_tnoise, hamiltonian)]).result()
    expval = results[0].data.evs
    # print(f"Results: {expval}")
    return expval


def get_single_out(initial_state, inst_number, echo, noise_model=None, include_x_terms=True):
    results = []
    for t in range(T):
        print(t)
        out = qc_qiskit(initial_state, L, g, hs[inst_number], phis[inst_number], t, int(L/2), echo=echo, noise_model=noise_model, include_x_terms=include_x_terms)
        results.append(out)
    results = np.array(results)
    return results.T



def get_instances(initial_state, echo, noise_model=None, include_x_terms=True):
    x_str = "with X" if include_x_terms else "without X"
    print(f"\nRunning {'echo' if echo else 'forward'} simulation {x_str} terms (single process)...")
    start_time = time.time()
    all_results = []
    for i in range(inst):
        print(f"Instance {i+1}/{inst} ({'echo' if echo else 'forward'}) {x_str}", end="\r")
        results = get_single_out(initial_state, i, echo, noise_model=noise_model, include_x_terms=include_x_terms)
        all_results.append(results)
    elapsed = time.time() - start_time
    print(f"\nCompleted {'echo' if echo else 'forward'} simulation {x_str} terms in {elapsed:.2f}s")
    all_results = np.array(all_results)
    return all_results

def savecsv(array, name):
    m,n,r = array.shape
    arr = np.column_stack((np.repeat(np.arange(m),n),array.reshape(m*n,-1)))
    df = pd.DataFrame(arr)
    df.to_csv(name)

energies_with_x = []
energies_without_x = []
# nprobs = np.arange(0, noise_prob+0.01, 0.05)
nprobs = [0, 0.001, 0.01, 0.1]
nprobs = [0.1]
state = initial_state

for nprob in nprobs:
    # Only add depolarizing error if not using backend noise model
    if not use_backend_noise:
        error = depolarizing_error(nprob, 1)
        noise_model.add_all_qubit_quantum_error(error, ["u1", "u2", "u3"],warnings=False)
    
    print(f"Running simulation for noise probability: {nprob}")
    
    # Run simulation with X terms
    energy_with_x = get_instances(state, echo=False, noise_model=noise_model, include_x_terms=True)
    av_energy_with_x = np.mean(energy_with_x, axis=0)
    energies_with_x.append(av_energy_with_x/L)
    
    # Run simulation without X terms
    energy_without_x = get_instances(state, echo=False, noise_model=noise_model, include_x_terms=False)
    av_energy_without_x = np.mean(energy_without_x, axis=0)
    energies_without_x.append(av_energy_without_x/L)
# autocorr_echo = get_instances(state, echo=True)
# av_autocorr_echo = np.mean(autocorr_echo, axis=0)

# Create DataFrame with energies organized by noise probability and X terms in columns
data = {'time': ts}
for i, nprob in enumerate(nprobs):
    data[f'energy_with_x_p_{nprob}'] = energies_with_x[i]
    data[f'energy_without_x_p_{nprob}'] = energies_without_x[i]

df = pd.DataFrame(data)
csv_filename = f"energy_comparison_{state}_g{g}_L{L}_inst{inst}_randomphi{args.randomphi}_delta{phi_delta}_amplitude{phi_amplitude}_noise{noise_prob}_usenoise{use_noise}.csv"

csv_path = f"{folder_name}/{csv_filename}"
df.to_csv(csv_path, index=False)
print(f"Energy data saved to {csv_path}")
print(f"Columns: {list(df.columns)}")

# Reshape data for seaborn relplot
# Convert from wide format to long format for both with and without X terms
df_melted = df.melt(id_vars=['time'], 
                    value_vars=[col for col in df.columns if col.startswith('energy_')],
                    var_name='config', 
                    value_name='energy')

# Extract X term configuration and noise probability from column names
df_melted['has_x_terms'] = df_melted['config'].str.contains('with_x')
df_melted['noise_prob'] = df_melted['config'].str.extract(r'p_([0-9.]+)').astype(float)
df_melted['x_label'] = df_melted['has_x_terms'].map({True: 'With X terms', False: 'Without X terms'})

# Function to find envelope of a signal
def find_envelope(signal, window_size=5):
    """Find the upper and lower envelope of a signal using rolling max/min with interpolation for smoothness"""
    from scipy import signal as scipy_signal
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

# Read autocorrelation data from existing CSV file
autocorr_folder = f"autocorr_data_L{L}_noiseprob{noise_prob}"
autocorr_filename = f"autocorr_data_{state}_g{g}_L{L}_inst{inst}_tf{args.tf}_randomphi{args.randomphi}_delta{phi_delta}_amplitude{phi_amplitude}_noise{noise_prob}_usenoise{use_noise}.csv"
autocorr_filepath = f"{autocorr_folder}/{autocorr_filename}"

try:
    autocorr_df = pd.read_csv(autocorr_filepath)
    print(f"Successfully loaded autocorrelation data from {autocorr_filepath}")
    autocorr_time = autocorr_df['time'].values
    autocorr_forward = autocorr_df['av_autocorr'].values
    autocorr_echo = autocorr_df['av_autocorr_echo'].values
    has_autocorr_data = True
except FileNotFoundError:
    print(f"Warning: Autocorrelation file not found at {autocorr_filepath}")
    has_autocorr_data = False

# Create a single plot comparing both Hamiltonians for one noise probability
fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))

# Get the noise probability (should be just one since nprobs = [0.1])
nprob = nprobs[0]

# Data with X terms
data_with_x = df_melted[(df_melted['noise_prob'] == nprob) & (df_melted['has_x_terms'] == True)]
time_vals_with_x = data_with_x['time'].values
energy_vals_with_x = data_with_x['energy'].values

# Data without X terms
data_without_x = df_melted[(df_melted['noise_prob'] == nprob) & (df_melted['has_x_terms'] == False)]
time_vals_without_x = data_without_x['time'].values
energy_vals_without_x = data_without_x['energy'].values

# Plot both energy series on the primary axis
ax1.plot(time_vals_with_x, energy_vals_with_x, label='With X terms (Full Hamiltonian)', 
        marker='o', markersize=4, linewidth=2, color='blue')
ax1.plot(time_vals_without_x, energy_vals_without_x, label='Without X terms (Z+ZZ only)', 
        marker='s', markersize=4, linewidth=2, linestyle='--', color='red')

# Add envelopes for better visualization
upper_env_with_x, lower_env_with_x = find_envelope(energy_vals_with_x, window_size=3)
upper_env_without_x, lower_env_without_x = find_envelope(energy_vals_without_x, window_size=3)

ax1.fill_between(time_vals_with_x, lower_env_with_x, upper_env_with_x, alpha=0.2, color='blue')
ax1.fill_between(time_vals_without_x, lower_env_without_x, upper_env_without_x, alpha=0.2, color='red')

ax1.set_xlabel('Time $t$', fontsize=14)
ax1.set_ylabel('Energy $E$', fontsize=14, color='black')
ax1.set_title(f'Hamiltonian Comparison & Autocorrelation: {state} state ($g={g}$, $L={L}$, $p={nprob}$)', fontsize=16)
ax1.tick_params(axis='y', labelcolor='black')
ax1.grid(True, alpha=0.3)

# Create secondary y-axis for autocorrelation data if available
if has_autocorr_data:
    ax2 = ax1.twinx()
    
    # Plot only echo autocorrelation data on secondary axis
    ax2.plot(autocorr_time, -autocorr_echo, label='-Echo', 
            marker='v', markersize=3, linewidth=1.5, color='black', alpha=0.8)
    
    # Find and plot envelope of echo data
    upper_env_echo, lower_env_echo = find_envelope(-autocorr_echo, window_size=5)
    ax2.fill_between(autocorr_time, lower_env_echo, upper_env_echo, alpha=0.5, color='orange')
    
    ax2.set_ylabel('- Echo', fontsize=14, color='black')
    ax2.tick_params(axis='y', labelcolor='black')

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
if has_autocorr_data:
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
else:
    ax1.legend(fontsize=12)

# Add text box with information about the Hamiltonians
# textstr = '\n'.join([
#     'Full Hamiltonian: $H = \\sum_i h_i Z_i + \\sum_i \\phi_i Z_i Z_{i+1} + g\\pi \\sum_i X_i$',
#     'Reduced Hamiltonian: $H = \\sum_i h_i Z_i + \\sum_i \\phi_i Z_i Z_{i+1}$'
# ])
# props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
# ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
#         verticalalignment='top', bbox=props)

plt.tight_layout()

# Save comprehensive data including energy and autocorrelation
if has_autocorr_data:
    # Create comprehensive DataFrame with all data
    comprehensive_data = {
        'time': ts,
        'energy_with_x': energies_with_x[0],  # Using first (and only) noise probability
        'energy_without_x': energies_without_x[0],
        'autocorr_forward': autocorr_forward[:len(ts)] if len(autocorr_forward) >= len(ts) else np.concatenate([autocorr_forward, np.nan * np.ones(len(ts) - len(autocorr_forward))]),
        'autocorr_echo': autocorr_echo[:len(ts)] if len(autocorr_echo) >= len(ts) else np.concatenate([autocorr_echo, np.nan * np.ones(len(ts) - len(autocorr_echo))]),
        'minus_autocorr_echo': -autocorr_echo[:len(ts)] if len(autocorr_echo) >= len(ts) else np.concatenate([-autocorr_echo, np.nan * np.ones(len(ts) - len(autocorr_echo))])
    }
    
    comprehensive_df = pd.DataFrame(comprehensive_data)
    comprehensive_csv_filename = f"comprehensive_data_{state}_g{g}_L{L}_inst{inst}_tf{args.tf}_randomphi{args.randomphi}_delta{phi_delta}_amplitude{phi_amplitude}_noise{noise_prob}_usenoise{use_noise}.csv"
    comprehensive_csv_path = f"{folder_name}/{comprehensive_csv_filename}"
    comprehensive_df.to_csv(comprehensive_csv_path, index=False)
    print(f"Comprehensive data (energy + autocorrelation) saved to {comprehensive_csv_path}")
    print(f"Comprehensive columns: {list(comprehensive_df.columns)}")
else:
    # Save only energy data if no autocorrelation data available
    comprehensive_data = {
        'time': ts,
        'energy_with_x': energies_with_x[0],
        'energy_without_x': energies_without_x[0]
    }
    
    comprehensive_df = pd.DataFrame(comprehensive_data)
    comprehensive_csv_filename = f"comprehensive_data_energy_only_{state}_g{g}_L{L}_inst{inst}_tf{args.tf}_randomphi{args.randomphi}_delta{phi_delta}_amplitude{phi_amplitude}_noise{noise_prob}_usenoise{use_noise}.csv"
    comprehensive_csv_path = f"{folder_name}/{comprehensive_csv_filename}"
    comprehensive_df.to_csv(comprehensive_csv_path, index=False)
    print(f"Comprehensive energy data saved to {comprehensive_csv_path}")
    print(f"Comprehensive columns: {list(comprehensive_df.columns)}")

# Save the comparison plot
plot_filename = f"hamiltonian_autocorr_comparison_{state}_g{g}_L{L}_inst{inst}_p{nprob}_tf{args.tf}.png"
plot_path = f"{folder_name}/{plot_filename}"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Hamiltonian and autocorrelation comparison plot saved to {plot_path}")

plt.show()