
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
parser.add_argument("--L", type=int, default=20, help="Number of qubits")
parser.add_argument("--device_name", type=int, default=0, help="Device name")
parser.add_argument("--inst", type=int, default=1, help="Number of instances for fig3d")
parser.add_argument("--randomphi", type=int, default=1, help="Prethermal=0 or DTC=1")
parser.add_argument("--phi_delta", type=float, default=0.0, help="Phi delta parameter")
parser.add_argument("--phi_amplitude", type=float, default=1.0, help="Phi amplitude parameter")
parser.add_argument("--tf",type=int, default=30, help="end time for fig3d")
parser.add_argument("--g",type=float, default=0.97, help="g for fig3d")
parser.add_argument("--noise_prob",type=float, default=0.05, help="noise probability")
parser.add_argument("--use_noise",type=int, default=1, help="use depolarizing noise: 0=no noise, 1=apply noise")
parser.add_argument("--initial_state",type=str, default="vacuum", help="initial state")
parser.add_argument("--use_fakebackend", type=int, default=0, help="Use FakeBackend for simulation: 0=No, 1=Yes")
parser.add_argument("--polarization", type=str, default="x", choices=["x", "y", "xy", "yx", "circular_left", "circular_right", "circular_static"], help="Polarization direction for rotations: x, y, xy, yx, circular_left, circular_right, or circular_static")
parser.add_argument("--circular_frequency", type=float, default=1.0, help="Frequency for time-dependent circular polarization")

args = parser.parse_args()

L = args.L
g = args.g
inst = args.inst
phi_delta = args.phi_delta
phi_amplitude = args.phi_amplitude
use_noise = args.use_noise
polarization = args.polarization
circular_frequency = args.circular_frequency

t_start = 0
t_end = args.tf
T = t_end - t_start
ts = np.arange(t_start, t_end, 1)
noise_prob = args.noise_prob
initial_state = args.initial_state


folder_name = f"autocorr_data_L{L}_circular-polarization"

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

def create_UF_subcircuit(L, g, phis, hs, polarization="x", time_step=0):
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
        elif polarization == "circular_left":
            # Left circular polarization: E_x = E₀cos(ωt), E_y = E₀sin(ωt)
            # Time-dependent circular polarization with adjustable frequency
            omega = circular_frequency  # Driving frequency
            angle_x = np.pi * g * np.cos(omega * time_step) / np.sqrt(2)
            angle_y = np.pi * g * np.sin(omega * time_step) / np.sqrt(2)
            subcircuit.rx(angle_x, i+1)  # X component (cosine)
            subcircuit.ry(angle_y, i+1)  # Y component (sine)
        elif polarization == "circular_right":
            # Right circular polarization: E_x = E₀cos(ωt), E_y = -E₀sin(ωt)
            omega = circular_frequency  # Driving frequency
            angle_x = np.pi * g * np.cos(omega * time_step) / np.sqrt(2)
            angle_y = -np.pi * g * np.sin(omega * time_step) / np.sqrt(2)
            subcircuit.rx(angle_x, i+1)  # X component (cosine)
            subcircuit.ry(angle_y, i+1)  # Y component (-sine)
        elif polarization == "circular_static":
            # Static circular polarization (your original approach)
            # Equal amplitude X and Y rotations with 90° phase difference
            subcircuit.rx(np.pi*g/np.sqrt(2), i+1)
            subcircuit.ry(np.pi*g/np.sqrt(2), i+1)
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
    
    # Forward evolution
    for time_step in range(t):
        UF_subcircuit = create_UF_subcircuit(L, g, phis, hs, polarization, time_step)
        circ.append(UF_subcircuit, range(L+1))
    
    # Echo evolution
    if echo:
        for time_step in range(t-1, -1, -1):  # Reverse time for echo
            UF_subcircuit = create_UF_subcircuit(L, g, phis, hs, polarization, time_step)
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
    # Save gate counts as CSV
    gate_count_filename = f"{folder_name}/gate_counts_t{t}_{echo_str}_opt{optimization_level}_{backend_name}_{coupling_str}_route{routing_method}_layout{layout_method}_polarization.csv"
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
        out = qc_qiskit(initial_state, L, g, hs[inst_number], phis[inst_number], t, int(L/2), polarization, echo=echo)
        results.append(out)
    results = np.array(results)
    return results.T



def get_instances(initial_state, echo):
    print(f"\nRunning {'echo' if echo else 'forward'} simulation (single process)...")
    start_time = time.time()
    all_results = []
    for i in range(inst):
        print(f"Instance {i+1}/{inst} ({'echo' if echo else 'forward'})")
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
    if len(peaks_max) >= 4:
        # Cubic spline interpolation for upper envelope (need at least 4 points for cubic)
        f_upper = interp1d(peaks_max, signal[peaks_max], kind='cubic', 
                          bounds_error=False, fill_value='extrapolate')
        upper_env = f_upper(time_indices)
    elif len(peaks_max) >= 2:
        # Linear interpolation if we have at least 2 points
        f_upper = interp1d(peaks_max, signal[peaks_max], kind='linear', 
                          bounds_error=False, fill_value='extrapolate')
        upper_env = f_upper(time_indices)
    else:
        # Fallback to constant if not enough points
        upper_env = np.full_like(signal, np.max(signal))
    
    if len(peaks_min) >= 4:
        # Cubic spline interpolation for lower envelope (need at least 4 points for cubic)
        f_lower = interp1d(peaks_min, signal[peaks_min], kind='cubic', 
                          bounds_error=False, fill_value='extrapolate')
        lower_env = f_lower(time_indices)
    elif len(peaks_min) >= 2:
        # Linear interpolation if we have at least 2 points
        f_lower = interp1d(peaks_min, signal[peaks_min], kind='linear', 
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
polarizations = ["x", "y", "circular_left", "circular_right"]
colors = ['#1f77b4', '#ff7f0e', '#9467bd', "#58b247"]  # Distinct colors
line_styles = ['-', '--', '-', '--']
markers = ['o', 's', 'v', '<']  # Different markers for each polarization
marker_sizes = [6, 6, 6, 6]

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.7*2, 4.3*2), sharex=True)
fig.suptitle(f"Polarization Comparison g={g}, L={L}, p={noise_prob}", fontsize=16)

all_data = {}

for i, pol in enumerate(polarizations):
    print(f"\n=== Running simulation for polarization: {pol} ===")
    
    # Run simulations for current polarization
    def get_instances_with_pol(initial_state, pol, echo):
        print(f"\nRunning {'echo' if echo else 'forward'} simulation for {pol} polarization...")
        start_time = time.time()
        all_results = []
        for j in range(inst):
            print(f"Instance {j+1}/{inst} ({'echo' if echo else 'forward'}) - {pol}")
            results = []
            for t in range(T):
                out = qc_qiskit(initial_state, L, g, hs[j], phis[j], t, int(L/2), pol, echo=echo)
                results.append(out)
            results = np.array(results)
            all_results.append(results.T)
        elapsed = time.time() - start_time
        print(f"\nCompleted {'echo' if echo else 'forward'} simulation for {pol} in {elapsed:.2f}s")
        all_results = np.array(all_results)
        return all_results
    
    autocorr = get_instances_with_pol(state, pol, echo=False)
    av_autocorr = np.mean(autocorr, axis=0)
    autocorr_echo = get_instances_with_pol(state, pol, echo=True)
    av_autocorr_echo = np.mean(autocorr_echo, axis=0)
    
    # Calculate envelopes for each signal
    forward_upper_env, forward_lower_env = find_envelope(av_autocorr, window_size=3)
    echo_upper_env, echo_lower_env = find_envelope(av_autocorr_echo, window_size=3)
    sqrt_echo_upper_env, sqrt_echo_lower_env = find_envelope(np.sqrt(av_autocorr_echo), window_size=3)
    
    # Store data for saving (including envelopes)
    all_data[pol] = {
        'time': ts,
        'av_autocorr': av_autocorr,
        'av_autocorr_echo': av_autocorr_echo,
        'sqrt_av_autocorr_echo': np.sqrt(av_autocorr_echo),
        'forward_upper_env': forward_upper_env,
        'forward_lower_env': forward_lower_env,
        'echo_upper_env': echo_upper_env,
        'echo_lower_env': echo_lower_env,
        'sqrt_echo_upper_env': sqrt_echo_upper_env,
        'sqrt_echo_lower_env': sqrt_echo_lower_env
    }
    
    # Save individual CSV for each polarization (with envelopes)
    df = pd.DataFrame(all_data[pol])
    csv_filename = f"autocorr_data_{state}_g{g}_L{L}_inst{inst}_randomphi{args.randomphi}_delta{phi_delta}_amplitude{phi_amplitude}_noise{noise_prob}_usenoise{use_noise}_pol{pol}_with_envelopes.csv"
    csv_path = f"{folder_name}/{csv_filename}"
    df.to_csv(csv_path, index=False)
    print(f"Autocorrelation data with envelopes for {pol} saved to {csv_path}")
    
    # Plot forward data on first subplot with envelopes
    ax1.plot(ts, av_autocorr, label=rf"$A$ - {pol.upper()}", color=colors[i], linestyle='-', 
            linewidth=3, marker=markers[i], markersize=6, alpha=0.9)
    ax1.fill_between(ts, forward_lower_env, forward_upper_env, alpha=0.15, color=colors[i], 
                    # label=f"Forward Envelope - {pol.upper()}"
                    )
    
    # Plot echo data on second subplot with envelopes
    ax2.plot(ts, av_autocorr_echo, label=rf"$A_0$ - {pol.upper()}", color=colors[i], linestyle='-', 
            linewidth=3, marker=markers[i], markersize=6, alpha=0.9)
    ax2.fill_between(ts, echo_lower_env, echo_upper_env, alpha=0.15, color=colors[i], 
                    # label=f"Echo Envelope - {pol.upper()}"
                    )
    ax2.plot(ts, np.sqrt(av_autocorr_echo), label=rf"$\sqrt{{A_0}}$ - {pol.upper()}", color=colors[i], linestyle='--', 
            linewidth=2.5, marker=markers[i], markersize=5, alpha=0.7)
    ax2.fill_between(ts, sqrt_echo_lower_env, sqrt_echo_upper_env, alpha=0.1, color=colors[i], 
                    # label=f"√Echo Envelope - {pol.upper()}"
                    )

# Configure first subplot (Forward Evolution)
ax1.set_ylabel(r"$\langle Z(0) Z(t) \rangle$", fontsize=12)
ax1.legend(fontsize=11, frameon=True, fancybox=True, shadow=True, loc='best')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_ylim(auto=True)
ax1.tick_params(axis='both', which='major', labelsize=12)

# Configure second subplot (Echo Evolution)
ax2.set_xlabel("t (FT)", fontsize=12)
ax2.set_ylabel(r"$\langle Z(0) Z(t) \rangle$", fontsize=12)
ax2.legend(fontsize=11, frameon=True, fancybox=True, shadow=True, loc='best', ncols=4)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_ylim(auto=True)
ax2.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()

# Save the plot with envelopes
plot_filename = f"autocorr_comparison_plot_{state}_g{g}_L{L}_inst{inst}_tf{args.tf}_with_envelopes.png"
plot_path = f"{folder_name}/{plot_filename}"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Comparison plot with envelopes saved to {plot_path}")
plt.xticks(ts)
plt.show()

# Save combined data (including envelopes)
combined_data = {'time': ts}
for pol in polarizations:
    combined_data[f'av_autocorr_{pol}'] = all_data[pol]['av_autocorr']
    combined_data[f'av_autocorr_echo_{pol}'] = all_data[pol]['av_autocorr_echo']
    combined_data[f'sqrt_av_autocorr_echo_{pol}'] = all_data[pol]['sqrt_av_autocorr_echo']
    combined_data[f'forward_upper_env_{pol}'] = all_data[pol]['forward_upper_env']
    combined_data[f'forward_lower_env_{pol}'] = all_data[pol]['forward_lower_env']
    combined_data[f'echo_upper_env_{pol}'] = all_data[pol]['echo_upper_env']
    combined_data[f'echo_lower_env_{pol}'] = all_data[pol]['echo_lower_env']
    combined_data[f'sqrt_echo_upper_env_{pol}'] = all_data[pol]['sqrt_echo_upper_env']
    combined_data[f'sqrt_echo_lower_env_{pol}'] = all_data[pol]['sqrt_echo_lower_env']

combined_df = pd.DataFrame(combined_data)
combined_csv_filename = f"autocorr_data_comparison_{state}_g{g}_L{L}_inst{inst}_randomphi{args.randomphi}_delta{phi_delta}_amplitude{phi_amplitude}_noise{noise_prob}_usenoise{use_noise}_with_envelopes.csv"
combined_csv_path = f"{folder_name}/{combined_csv_filename}"
combined_df.to_csv(combined_csv_path, index=False)
print(f"Combined autocorrelation comparison data with envelopes saved to {combined_csv_path}")
