
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
parser.add_argument("--tf",type=int, default=20, help="end time for fig3d")
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


folder_name = f"energy-data_L{L}-full-ham"

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
 



def get_hamiltonian(L, g, phis, hs, instance=0):
    ham = []
    z_str = "I" * (L)
    zz_str = "I" * (L)
    x_str = "I" * (L)
    
    for i in range(L):
        z = z_str[:i] + 'Z' + z_str[i+1:]
        ham.append((z, hs[i]))

    for i in range(L-1):
        zz = zz_str[:i] + 'ZZ' + zz_str[i+2:]
        ham.append((zz, phis[i]))

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


def qc_qiskit(initial_state, L, g, hs, phis, t, qubit, echo=False, noise_model=None):
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

    
    if args.use_fakebackend == 1:
        backend = FakeBrisbane()
        print("Using FakeBrisbane backend.\n")
    else:
        
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
    hamiltonian = get_hamiltonian(L, g, phis, hs)
    results = estimator.run([(circ_tnoise, hamiltonian)]).result()
    expval = results[0].data.evs
    # print(f"Results: {expval}")
    return expval


def get_single_out(initial_state, inst_number, echo, noise_model=None):
    results = []
    for t in range(T):
        print(t)
        out = qc_qiskit(initial_state, L, g, hs[inst_number], phis[inst_number], t, int(L/2), echo=echo, noise_model=noise_model)
        results.append(out)
    results = np.array(results)
    return results.T



def get_instances(initial_state, echo, noise_model=None):
    print(f"\nRunning {'echo' if echo else 'forward'} simulation (single process)...")
    start_time = time.time()
    all_results = []
    for i in range(inst):
        print(f"Instance {i+1}/{inst} ({'echo' if echo else 'forward'})", end="\r")
        results = get_single_out(initial_state, i, echo, noise_model=noise_model)
        all_results.append(results)
    elapsed = time.time() - start_time
    print(f"\nCompleted {'echo' if echo else 'forward'} simulation in {elapsed:.2f}s")
    all_results = np.array(all_results)
    return all_results

def savecsv(array, name):
    m,n,r = array.shape
    arr = np.column_stack((np.repeat(np.arange(m),n),array.reshape(m*n,-1)))
    df = pd.DataFrame(arr)
    df.to_csv(name)

energies = []
# nprobs = np.arange(0, noise_prob+0.01, 0.05)
nprobs = [0, 0.001, 0.01, 0.1]
# nprobs = [0.1]
state = initial_state
for nprob in nprobs:
    error = depolarizing_error(nprob, 1)
    noise_model.add_all_qubit_quantum_error(error, ["u1", "u2", "u3"],warnings=False)
    
    print(f"Running simulation for noise probability: {nprob}")
    energy = get_instances(state, echo=False, noise_model=noise_model)
    av_energy = np.mean(energy, axis=0)
    energies.append(av_energy/L)
# autocorr_echo = get_instances(state, echo=True)
# av_autocorr_echo = np.mean(autocorr_echo, axis=0)

# Create DataFrame with energies organized by noise probability in columns
data = {'time': ts}
for i, nprob in enumerate(nprobs):
    data[f'energy_p_{nprob}'] = energies[i]

df = pd.DataFrame(data)
csv_filename = f"energy_data_{state}_g{g}_L{L}_inst{inst}_randomphi{args.randomphi}_delta{phi_delta}_amplitude{phi_amplitude}_noise{noise_prob}_usenoise{use_noise}.csv"

csv_path = f"{folder_name}/{csv_filename}"
df.to_csv(csv_path, index=False)
print(f"Energy data saved to {csv_path}")
print(f"Columns: {list(df.columns)}")

# Reshape data for seaborn relplot
# Convert from wide format to long format
df_melted = df.melt(id_vars=['time'], 
                    value_vars=[col for col in df.columns if col.startswith('energy_')],
                    var_name='noise_prob', 
                    value_name='energy')

# Extract noise probability from column names for better labeling
df_melted['noise_prob'] = df_melted['noise_prob'].str.replace('energy_p_', '').astype(float)

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

# Create subplots for better visualization
fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))

# Plot: Energy vs time with envelopes for each noise probability
for nprob in df_melted['noise_prob'].unique():
    data_subset = df_melted[df_melted['noise_prob'] == nprob]
    time_vals = data_subset['time'].values
    energy_vals = data_subset['energy'].values
    
    # Plot the original signal
    ax1.plot(time_vals, energy_vals, label=f'p={nprob}', marker='o', markersize=3)
    
    # Find and plot envelopes
    upper_env, lower_env = find_envelope(energy_vals, window_size=3)
    ax1.fill_between(time_vals, lower_env, upper_env, alpha=0.2)

ax1.set_xlabel('Time $t$')
ax1.set_ylabel('Energy $E$')
ax1.set_title(f'Energy with Envelopes for {state} state ($g={g}$, $L={L}$)')
ax1.legend(title='Noise Probability $p$')
ax1.grid(True, alpha=0.3)

plt.tight_layout()

# Customize the plot
plt.title(f'Energy Analysis for {state} state ($g={g}$, $L={L}$)')

# Save the plot
plot_filename = f"energy_plot_{state}_g{g}_L{L}_inst{inst}_tf{args.tf}.png"
plot_path = f"{folder_name}/{plot_filename}"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {plot_path}")

plt.show()