
import numpy as np
import matplotlib.pyplot as plt
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
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


parser = argparse.ArgumentParser(description="Plot figures")
parser.add_argument("--L", type=int, default=4, help="Number of qubits")
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
parser.add_argument("--polarization", type=str, default="x", choices=["x", "y", "xy", "yx"], help="Polarization direction for rotations: x, y, xy, or yx")

args = parser.parse_args()
color_palette = ["#361AC1", "#15B300", "#008DBC", "#DF349E", "#0C8BCA", "#FF9100", "#E72142", '#AA4499']

L = args.L
g = args.g
inst = args.inst
phi_delta = args.phi_delta
phi_amplitude = args.phi_amplitude
use_noise = args.use_noise
polarization = args.polarization

t_start = 0
t_end = args.tf
T = t_end - t_start
ts = np.arange(t_start, t_end, 1)
noise_prob = args.noise_prob
initial_state = args.initial_state


folder_name = f"autocorr_data_L{L}_polarization_xy_cycle"

if not os.path.exists(folder_name):
    os.makedirs(folder_name)


# Load hs and phis from CSV files
disorder_folder = "disorder_data"
hs_filename = os.path.join(disorder_folder, f"hs_L{L}_inst{100}_ampl{phi_amplitude}_delta{phi_delta}_randomphi{args.randomphi}.csv")
phis_filename = os.path.join(disorder_folder, f"phis_L{L}_inst{100}_ampl{phi_amplitude}_delta{phi_delta}_randomphi{args.randomphi}.csv")

# Use .iloc to select first 'inst' rows
hs_df = pd.read_csv(hs_filename, comment='#', header=0)
phis_df = pd.read_csv(phis_filename, comment='#', header=0)
hs = hs_df.iloc[:inst].values
phis = phis_df.iloc[:inst].values

noise_model = NoiseModel()
 
# Add depolarizing error to all single qubit u1, u2, u3 gates
error = depolarizing_error(noise_prob, 1)
noise_model.add_all_qubit_quantum_error(error, ["u1", "u2", "u3"],warnings=False)


# Setup Qiskit backend and noise model
# backend = Aer.get_backend('aer_simulator',noise_model=noise_model)

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

def create_UF_subcircuit(L, g, phis, hs, polarization="x"):
    subcircuit = QuantumCircuit(L+1)
    for i in range(L):
        if polarization == "x":
            subcircuit.rx(np.pi*g, i+1)
        elif polarization == "y":
            subcircuit.ry(np.pi*g, i+1)
        elif polarization == "xy":
            subcircuit.rx(np.pi*g/2, i+1)
            subcircuit.ry(np.pi*g/2, i+1)
        elif polarization == "yx":
            subcircuit.ry(np.pi*g/2, i+1)
            subcircuit.rx(np.pi*g/2, i+1)
    for i in range(0, L-1, 2):
        subcircuit.rzz(phis[i], i+1, i+2)
    for i in range(1, L-1, 2):
        subcircuit.rzz(phis[i], i+1, i+2)
    for i in range(L):
        subcircuit.rz(hs[i], i+1)
    return subcircuit


def qc_qiskit(initial_state, L, g, hs, phis, t, qubit, polarization="x", echo=False):
    circ = QuantumCircuit(L+1,1)
    # Initial state
    if initial_state == "neel":
        for i in range(1, L+1):
            if i % 2 == 0:
                circ.x(i)
    # Hadamard and entangle
    circ.h(0)
    circ.cz(qubit+1, 0)
    
    # Forward evolution with alternating polarization every 5 time steps
    # Pattern: X for steps 0-4, Y for steps 5-9, X for steps 10-14, Y for steps 15-19, etc.
    for step in range(t):
        # Determine polarization based on time step (every 5 seconds)
        current_polarization = "x" if (step // 5) % 2 == 0 else "y"
        UF_subcircuit = create_UF_subcircuit(L, g, phis, hs, current_polarization)
        circ.append(UF_subcircuit, range(L+1))
    
    # Echo evolution with same alternating pattern (but in reverse)
    if echo:
        for step in range(t-1, -1, -1):  # Reverse order for echo
            current_polarization = "x" if (step // 5) % 2 == 0 else "y"
            UF_subcircuit = create_UF_subcircuit(L, g, phis, hs, current_polarization)
            UF_inv = UF_subcircuit.inverse()
            circ.append(UF_inv, range(L+1))

    circ.cz(qubit+1, 0)
    circ.h(0)
    circ.measure(0,0)

    # circ.draw("mpl")
    # plt.show()
    # Select backend
    if args.use_fakebackend == 1:
        backend = FakeBrisbane()
        # print("Using FakeBrisbane backend.")
    else:
        backend = AerSimulator(noise_model=noise_model, device="GPU",cuStateVec_enable=True)
    # Transpile circuit for noisy basis gates
    # Define a linear coupling map for L+1 qubits
    coupling_list = [(i, i+1) for i in range(1,L)] + [(0, int(L/2))]
    # print(coupling_list)
    # print(dict(coupling_list))
    coupling_map = CouplingMap(couplinglist=coupling_list)
    coords = [(i, i**2/10) for i in range(L+1)]
    # fig = plot_coupling_map(L+1, coords, coupling_list)
    # coupling_map_filename = f"{folder_name}/coupling_map_L{L+1}.png"
    # fig.savefig(coupling_map_filename)
    # plt.close(fig)

    use_coupling = len(coupling_list) > 0
    coupling_str = "coupling" if use_coupling else "nocoupling"
    optimization_level = 0
    routing_method = "lookahead"  # You can change to "lookahead", "sabre", etc.
    layout_method = "dense"
    # Make initial layout follow the coupling map
    # Use the first L+1 unique nodes from the coupling map
    circ_qubits = [circ.qubits[i] for i in range(L+1)]
    snake_layout = [15,30,17,12,11,10,9,8,7,6,5,4,3,2,1,0,14,18,19,20,21]
    layout_dict = {circ_qubits[i]: snake_layout[i] for i in range(L+1)}
    initial_layout = Layout(layout_dict)
    # print(f"Initial layout: {initial_layout}")
    passmanager = generate_preset_pass_manager(
        optimization_level=optimization_level,
        backend=backend,
        # coupling_map=coupling_map,
        # routing_method=routing_method,
        routing_method=None,
        initial_layout=initial_layout,
        # layout_method=layout_method  # You can change to "trivial", "dense
    )
    echo_str = "echo" if echo else "forward"
    backend_name = getattr(backend, 'name', str(type(backend).__name__))
    circ_tnoise = passmanager.run(circ)
    gate_counts = circ_tnoise.count_ops()
    # print(gate_counts)
    # Save gate counts as CSV (updated filename for alternating polarization)
    gate_count_filename = f"{folder_name}/gate_counts_t{t}_{echo_str}_opt{optimization_level}_{backend_name}_{coupling_str}_route{routing_method}_layout{layout_method}_alternating_xy_5s.csv"
    pd.DataFrame(list(gate_counts.items()), columns=["gate", "count"]).to_csv(gate_count_filename, index=False)
    
    # layout_fig = plot_circuit_layout(circ_tnoise, backend=backend)
    # layout_str = "-".join(str(q) for q in initial_layout)
    # layout_filename = f"{folder_name}/circuit_layout_t{t}_{echo_str}_opt{optimization_level}_{backend_name}_{coupling_str}_route{routing_method}_layout{layout_method}.png"
    # layout_fig.savefig(layout_filename)
    # plt.close(layout_fig)

    # # Draw and save the transpiled circuit
    # circuit_fig = circ_tnoise.draw(output='mpl')
    # circuit_filename = f"{folder_name}/transpiled_circuit_t{t}_{echo_str}_opt{optimization_level}_{backend_name}_{coupling_str}_route{routing_method}_layout{layout_method}.png"
    # circuit_fig.savefig(circuit_filename)
    # plt.close(circuit_fig)
    # Run and get counts
    result = backend.run(circ_tnoise).result()
    counts = result.get_counts(circ_tnoise)
    expval = compute_z_expectation(counts, 1)[0]
    return expval


def get_single_out(initial_state, inst_number, echo):
    results = []
    for t in range(T):
        print(f"time: {t}")
        out = qc_qiskit(initial_state, L, g, hs[inst_number], phis[inst_number], t, int(L/2), echo=echo)
        results.append(out)
    results = np.array(results)
    return results.T



def get_instances(initial_state, echo):
    print(f"\nRunning {'echo' if echo else 'forward'} simulation (single process)...")
    start_time = time.time()
    all_results = []
    for i in range(inst):
        print(f"Instance {i+1}/{inst} ({'echo' if echo else 'forward'})", end="\r")
        results = get_single_out(initial_state, i, echo)
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

def find_envelope(signal, window_size=5):
    """Find the upper and lower envelope of a signal using rolling max/min with interpolation for smoothness"""
    
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
    sigma = max(0.5, window_size/4)  # Adaptive smoothing based on window size
    upper_env = gaussian_filter1d(upper_env, sigma=sigma)
    lower_env = gaussian_filter1d(lower_env, sigma=sigma)
    
    # Final check to ensure envelopes bound the signal properly
    upper_env = np.maximum(upper_env, signal)
    lower_env = np.minimum(lower_env, signal)
    
    return upper_env, lower_env

# state = np.random.random(2**L)
# state = state / np.linalg.norm(state)  # normalize the state
# state = np.zeros(2**L)
# state[0] = 1 

# state = np.random.random(2**L)
# state = state / np.linalg.norm(state)  # normalize the state
# state = np.zeros(2**L)
# state[0] = 1 

state = initial_state

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.7*2, 4.3*2))
fig.suptitle(f"Alternating X-Y Polarization (every 5 FT) g={g}, L={L}, noise={noise_prob}", fontsize=12)

print(f"\n=== Running simulation with alternating X-Y polarization every 5 seconds ===")

# Run simulations with alternating polarization
autocorr = get_instances(state, echo=False)
av_autocorr = np.mean(autocorr, axis=0)
autocorr_echo = get_instances(state, echo=True)
av_autocorr_echo = np.mean(autocorr_echo, axis=0)

# Store data for saving
all_data = {
    'time': ts,
    'av_autocorr': av_autocorr,
    'av_autocorr_echo': av_autocorr_echo,
    'sqrt_av_autocorr_echo': np.sqrt(av_autocorr_echo)
}

# Calculate envelopes for autocorrelation data
forward_upper_env, forward_lower_env = find_envelope(av_autocorr, window_size=3)
echo_upper_env, echo_lower_env = find_envelope(av_autocorr_echo, window_size=3)
sqrt_echo_upper_env, sqrt_echo_lower_env = find_envelope(np.sqrt(av_autocorr_echo), window_size=3)

# Add envelope data to the dictionary
all_data.update({
    'forward_upper_env': forward_upper_env,
    'forward_lower_env': forward_lower_env,
    'echo_upper_env': echo_upper_env,
    'echo_lower_env': echo_lower_env,
    'sqrt_echo_upper_env': sqrt_echo_upper_env,
    'sqrt_echo_lower_env': sqrt_echo_lower_env
})

# Save CSV data
df = pd.DataFrame(all_data)
csv_filename = f"autocorr_data_{state}_g{g}_L{L}_inst{inst}_randomphi{args.randomphi}_delta{phi_delta}_amplitude{phi_amplitude}_noise{noise_prob}_usenoise{use_noise}_alternating_xy_5s_with_envelopes.csv"
csv_path = f"{folder_name}/{csv_filename}"
df.to_csv(csv_path, index=False)
print(f"Autocorrelation data with envelopes saved to {csv_path}")

# Plot forward data on first subplot with envelopes
ax1.plot(ts, av_autocorr, label=r"$\bar{A}$", color=color_palette[0], linestyle='-', 
        linewidth=3, marker='o', markersize=6, alpha=0.9)
ax1.fill_between(ts, forward_lower_env, forward_upper_env, alpha=0.2, color=color_palette[0], 
                # label="Forward Envelope"
                )

# Plot echo data on second subplot with envelopes
ax2.plot(ts, av_autocorr_echo, label=r"$\bar{A_0}$", color=color_palette[1], linestyle='-', 
        linewidth=3, marker='o', markersize=6, alpha=0.9)
ax2.fill_between(ts, echo_lower_env, echo_upper_env, alpha=0.2, color=color_palette[1], 
                # label="Echo Envelope"
                )
ax2.plot(ts, np.sqrt(av_autocorr_echo), label=r"$\sqrt{\bar{A_0}}$", color=color_palette[2], linestyle='--', 
        linewidth=2.5, marker='s', markersize=5, alpha=0.7)
ax2.fill_between(ts, sqrt_echo_lower_env, sqrt_echo_upper_env, alpha=0.15, color=color_palette[2], 
                # label="√Echo Envelope"
                )

# Add vertical lines to indicate polarization changes (every 5 time steps)
for t_change in range(5, t_end, 5):
    pol_label = "Y" if ((t_change // 5) % 2 == 1) else "X"
    ax1.axvline(x=t_change, color='red', linestyle=':', alpha=0.5)
    ax2.axvline(x=t_change, color='red', linestyle=':', alpha=0.5)
    # if t_change <= 15:  # Only label first few changes to avoid clutter
    ax1.text(t_change+0.1, ax1.get_ylim()[1]*0.9, f"→{pol_label}", rotation=90, fontsize=10, color='red')

# Configure first subplot (Forward Evolution)
ax1.set_ylabel(r"$\langle Z(0) Z(t) \rangle$", fontsize=12,)
ax1.legend(fontsize=11, frameon=True, fancybox=True, shadow=True, loc='best')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_ylim(auto=True)
ax1.tick_params(axis='both', which='major', labelsize=12)

# Configure second subplot (Echo Evolution)
ax2.set_xlabel("t (FT)", fontsize=14,)
ax2.set_ylabel(r"$\langle Z(0) Z(t) \rangle$", fontsize=12,)
ax2.legend(fontsize=11, frameon=True, fancybox=True, shadow=True, loc='best')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_ylim(auto=True)
ax2.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()

# Save the plot with envelopes
plot_filename = f"autocorr_plot_{state}_g{g}_L{L}_inst{inst}_tf{args.tf}_alternating_xy_5s_with_envelopes.png"
plot_path = f"{folder_name}/{plot_filename}"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Plot with envelopes saved to {plot_path}")

plt.show()
