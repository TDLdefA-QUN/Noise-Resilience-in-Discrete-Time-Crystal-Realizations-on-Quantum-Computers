
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
parser.add_argument("--L", type=int, default=20, help="Number of qubits")
parser.add_argument("--device_name", type=int, default=0, help="Device name")
parser.add_argument("--inst", type=int, default=1, help="Number of instances for fig3d")
parser.add_argument("--randomphi", type=int, default=1, help="Prethermal=0 or DTC=1")
parser.add_argument("--phi_delta", type=float, default=0.0, help="Phi delta parameter")
parser.add_argument("--phi_amplitude", type=float, default=1.0, help="Phi amplitude parameter")
parser.add_argument("--tf",type=int, default=50, help="end time for fig3d")
parser.add_argument("--g",type=float, default=0.97, help="g for fig3d")
parser.add_argument("--noise_prob",type=float, default=0.05, help="noise probability")
parser.add_argument("--use_noise",type=int, default=1, help="use depolarizing noise: 0=no noise, 1=apply noise")
parser.add_argument("--initial_state",type=str, default="vacuum", help="initial state")
parser.add_argument("--use_fakebackend", type=int, default=0, help="Use FakeBackend for simulation: 0=No, 1=Yes")

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

# Initialize noise model based on use_fakebackend setting
if args.use_fakebackend == 1:
    # When using fake backend, the backend itself provides the noise model
    noise_model = NoiseModel.from_backend(FakeBrisbane())
else:
    # When using AerSimulator, create a custom noise model
    noise_model = NoiseModel()
 



def get_hamiltonian(L, g, phis, hs, instance=0, hamiltonian_type="full"):
    """
    Create Hamiltonian based on specified type:
    - "full": Z + ZZ + X terms
    - "z_only": Only Z terms (disorder)
    - "zz_only": Only ZZ terms (interactions)
    - "x_only": Only X terms (transverse field)
    - "z_zz": Z + ZZ terms (no X)
    """
    ham = []
    z_str = "I" * (L)
    zz_str = "I" * (L)
    x_str = "I" * (L)
    
    # Z terms (disorder)
    if hamiltonian_type in ["full", "z_only", "z_zz"]:
        for i in range(L):
            z = z_str[:i] + 'Z' + z_str[i+1:]
            ham.append((z, hs[i]))

    # ZZ terms (interaction)
    if hamiltonian_type in ["full", "zz_only", "z_zz"]:
        for i in range(L-1):
            zz = zz_str[:i] + 'ZZ' + zz_str[i+2:]
            ham.append((zz, phis[i]))

    # X terms (transverse field)
    if hamiltonian_type in ["full", "x_only"]:
        for i in range(L):
            x = x_str[:i] + 'X' + x_str[i+1:]
            ham.append((x, g*np.pi))
    
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


def qc_qiskit(initial_state, L, g, hs, phis, t, qubit, echo=False, noise_model=None, hamiltonian_type="full"):
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
    backend = AerSimulator(noise_model=noise_model, device="GPU", cuStateVec_enable=True)
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
    hamiltonian = get_hamiltonian(L, g, phis, hs, hamiltonian_type=hamiltonian_type)
    results = estimator.run([(circ_tnoise, hamiltonian)]).result()
    expval = results[0].data.evs
    # print(f"Results: {expval}")
    return expval


def get_single_out(initial_state, inst_number, echo, noise_model=None, hamiltonian_type="full"):
    results = []
    for t in range(T):
        print(t)
        out = qc_qiskit(initial_state, L, g, hs[inst_number], phis[inst_number], t, int(L/2), echo=echo, noise_model=noise_model, hamiltonian_type=hamiltonian_type)
        results.append(out)
    results = np.array(results)
    return results.T



def get_instances(initial_state, echo, noise_model=None, hamiltonian_type="full"):
    ham_type_str = hamiltonian_type.replace("_", " ").title()
    print(f"\nRunning {'echo' if echo else 'forward'} simulation with {ham_type_str} Hamiltonian (single process)...")
    start_time = time.time()
    all_results = []
    for i in range(inst):
        print(f"Instance {i+1}/{inst} ({'echo' if echo else 'forward'}) {ham_type_str}", end="\r")
        results = get_single_out(initial_state, i, echo, noise_model=noise_model, hamiltonian_type=hamiltonian_type)
        all_results.append(results)
    elapsed = time.time() - start_time
    print(f"\nCompleted {'echo' if echo else 'forward'} simulation with {ham_type_str} Hamiltonian in {elapsed:.2f}s")
    all_results = np.array(all_results)
    return all_results

def savecsv(array, name):
    m,n,r = array.shape
    arr = np.column_stack((np.repeat(np.arange(m),n),array.reshape(m*n,-1)))
    df = pd.DataFrame(arr)
    df.to_csv(name)

energies_z_only = []
energies_zz_only = []
energies_x_only = []
energies_sum = []
energies_full = []
# nprobs = np.arange(0, noise_prob+0.01, 0.05)
nprobs = [0, 0.001, 0.01, 0.1]
nprobs = [noise_prob]
state = initial_state

for nprob in nprobs:
    # Only add depolarizing error when using AerSimulator (not fake backend)
    if args.use_fakebackend == 0:
        error = depolarizing_error(nprob, 1)
        noise_model.add_all_qubit_quantum_error(error, ["u1", "u2", "u3"], warnings=False)
        print(f"Running simulation with custom depolarizing noise (p={nprob})")
    else:
        print(f"Running simulation with FakeBrisbane noise model (ignoring nprob={nprob})")
    
    # Run simulation with only Z terms
    energy_z_only = get_instances(state, echo=False, noise_model=noise_model, hamiltonian_type="z_only")
    av_energy_z_only = np.mean(energy_z_only, axis=0)
    energies_z_only.append(av_energy_z_only/L)
    
    # Run simulation with only ZZ terms
    energy_zz_only = get_instances(state, echo=False, noise_model=noise_model, hamiltonian_type="zz_only")
    av_energy_zz_only = np.mean(energy_zz_only, axis=0)
    energies_zz_only.append(av_energy_zz_only/L)
    
    # Run simulation with only X terms
    energy_x_only = get_instances(state, echo=False, noise_model=noise_model, hamiltonian_type="x_only")
    av_energy_x_only = np.mean(energy_x_only, axis=0)
    energies_x_only.append(av_energy_x_only/L)
    
    # Calculate sum of Z and ZZ energies
    av_energy_sum = av_energy_z_only + av_energy_zz_only
    energies_sum.append(av_energy_sum/L)
    
    # Run simulation with full Hamiltonian (Z + ZZ + X)
    energy_full = get_instances(state, echo=False, noise_model=noise_model, hamiltonian_type="full")
    av_energy_full = np.mean(energy_full, axis=0)
    energies_full.append(av_energy_full/L)
# autocorr_echo = get_instances(state, echo=True)
# av_autocorr_echo = np.mean(autocorr_echo, axis=0)

# Create DataFrame with energies organized by noise probability and Hamiltonian types
data = {'time': ts}
for i, nprob in enumerate(nprobs):
    data[f'energy_z_only_p_{nprob}'] = energies_z_only[i]
    data[f'energy_zz_only_p_{nprob}'] = energies_zz_only[i]
    data[f'energy_x_only_p_{nprob}'] = energies_x_only[i]
    data[f'energy_sum_p_{nprob}'] = energies_sum[i]
    data[f'energy_full_p_{nprob}'] = energies_full[i]

df = pd.DataFrame(data)
csv_filename = f"energy_comparison_all_{state}_g{g}_L{L}_inst{inst}_randomphi{args.randomphi}_delta{phi_delta}_amplitude{phi_amplitude}_noise{noise_prob}_usenoise{use_noise}.csv"

csv_path = f"{folder_name}/{csv_filename}"
df.to_csv(csv_path, index=False)
print(f"Energy data saved to {csv_path}")
print(f"Columns: {list(df.columns)}")

# Reshape data for seaborn relplot
# Convert from wide format to long format for all three Hamiltonian types
df_melted = df.melt(id_vars=['time'], 
                    value_vars=[col for col in df.columns if col.startswith('energy_')],
                    var_name='config', 
                    value_name='energy')

# Extract Hamiltonian type and noise probability from column names
df_melted['hamiltonian_type'] = df_melted['config'].str.extract(r'energy_([^_]+(?:_[^_]+)?)_p_')[0]
df_melted['noise_prob'] = df_melted['config'].str.extract(r'p_([0-9.]+)').astype(float)
df_melted['ham_label'] = df_melted['hamiltonian_type'].map({
    'z_only': 'Z terms only', 
    'zz_only': 'ZZ terms only',
    'x_only': 'X terms only',
    'sum': 'Z + ZZ (sum)',
    'full': 'Z + ZZ + X (full)'
})

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

# Create a single plot comparing all five Hamiltonian types for one noise probability
fig, ax = plt.subplots(1, 1, figsize=(16, 10))

# Get the noise probability (should be just one since nprobs = [noise_prob])
nprob = nprobs[0]

# Data for Z terms only
data_z_only = df_melted[(df_melted['noise_prob'] == nprob) & (df_melted['hamiltonian_type'] == 'z_only')]
time_vals_z = data_z_only['time'].values
energy_vals_z = data_z_only['energy'].values

# Data for ZZ terms only
data_zz_only = df_melted[(df_melted['noise_prob'] == nprob) & (df_melted['hamiltonian_type'] == 'zz_only')]
time_vals_zz = data_zz_only['time'].values
energy_vals_zz = data_zz_only['energy'].values

# Data for X terms only
data_x_only = df_melted[(df_melted['noise_prob'] == nprob) & (df_melted['hamiltonian_type'] == 'x_only')]
time_vals_x = data_x_only['time'].values
energy_vals_x = data_x_only['energy'].values

# Data for sum (Z + ZZ)
data_sum = df_melted[(df_melted['noise_prob'] == nprob) & (df_melted['hamiltonian_type'] == 'sum')]
time_vals_sum = data_sum['time'].values
energy_vals_sum = data_sum['energy'].values

# Data for full Hamiltonian (Z + ZZ + X)
data_full = df_melted[(df_melted['noise_prob'] == nprob) & (df_melted['hamiltonian_type'] == 'full')]
time_vals_full = data_full['time'].values
energy_vals_full = data_full['energy'].values

# Plot all five series
ax.plot(time_vals_z, energy_vals_z, label='Z terms only ($\\sum_i h_i Z_i$)', 
        marker='o', markersize=3, linewidth=2, color='blue')
ax.plot(time_vals_zz, energy_vals_zz, label='ZZ terms only ($\\sum_i \\phi_i Z_i Z_{i+1}$)', 
        marker='s', markersize=3, linewidth=2, linestyle='--', color='red')
ax.plot(time_vals_x, energy_vals_x, label='X terms only ($g\\pi \\sum_i X_i$)', 
        marker='d', markersize=3, linewidth=2, linestyle=':', color='orange')
ax.plot(time_vals_sum, energy_vals_sum, label='Z + ZZ (sum)', 
        marker='^', markersize=3, linewidth=2, linestyle='-.', color='green')
ax.plot(time_vals_full, energy_vals_full, label='Z + ZZ + X (full)', 
        marker='v', markersize=3, linewidth=3, linestyle='-', color='purple')

# Add envelopes for better visualization
upper_env_z, lower_env_z = find_envelope(energy_vals_z, window_size=3)
upper_env_zz, lower_env_zz = find_envelope(energy_vals_zz, window_size=3)
upper_env_x, lower_env_x = find_envelope(energy_vals_x, window_size=3)
upper_env_sum, lower_env_sum = find_envelope(energy_vals_sum, window_size=3)
upper_env_full, lower_env_full = find_envelope(energy_vals_full, window_size=3)

ax.fill_between(time_vals_z, lower_env_z, upper_env_z, alpha=0.15, color='blue')
ax.fill_between(time_vals_zz, lower_env_zz, upper_env_zz, alpha=0.15, color='red')
ax.fill_between(time_vals_x, lower_env_x, upper_env_x, alpha=0.15, color='orange')
ax.fill_between(time_vals_sum, lower_env_sum, upper_env_sum, alpha=0.15, color='green')
ax.fill_between(time_vals_full, lower_env_full, upper_env_full, alpha=0.15, color='purple')

ax.set_xlabel('Time $t$', fontsize=14)
ax.set_ylabel('Energy $E$', fontsize=14)
ax.set_title(f'Complete Hamiltonian Components Comparison: {state} state ($g={g}$, $L={L}$, $p={nprob}$)', fontsize=16)
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()

# Save the comparison plot
plot_filename = f"hamiltonian_all_comparison_{state}_g{g}_L{L}_inst{inst}_p{nprob}_tf{args.tf}.png"
plot_path = f"{folder_name}/{plot_filename}"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Complete Hamiltonian comparison plot saved to {plot_path}")

plt.show()