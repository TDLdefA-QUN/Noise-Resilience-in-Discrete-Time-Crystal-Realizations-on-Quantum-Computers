
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import pandas as pd
from functools import partial
from scipy.optimize import curve_fit
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

parser = argparse.ArgumentParser(description="Plot figures")
parser.add_argument("--L", type=int, default=20, help="Number of qubits")
parser.add_argument("--device_name", type=int, default=0, help="Device name")
parser.add_argument("--inst", type=int, default=10, help="Number of instances for fig3d")
parser.add_argument("--randomphi", type=int, default=1, help="Prethermal=0 or DTC=1")
parser.add_argument("--phi_delta", type=float, default=0.0, help="Phi delta parameter")
parser.add_argument("--phi_amplitude", type=float, default=1.0, help="Phi amplitude parameter")
parser.add_argument("--tf",type=int, default=20, help="end time for fig3d")
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


folder_name = f"energy-data_L{L}-fakebrisbane"

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
    brisbane_nqubit = 127
    z_str = "I" * (brisbane_nqubit)
    zz_str = "I" * (brisbane_nqubit)
    x_str = "I" * (brisbane_nqubit)

    for i in range(L):
        z = z_str[:i] + 'Z' + z_str[i+1:]
        ham.append((z, hs[i]))

    for i in range(L-1):
        zz = zz_str[:i] + 'ZZ' + zz_str[i+2:]
        ham.append((zz, phis[i]))
    # for i in range(L):
    #     z = x_str[:i] + 'X' + x_str[i+1:]
    #     ham.append((z, g*np.pi))
    
    hamiltonian = SparsePauliOp.from_list(ham, num_qubits=brisbane_nqubit)
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
        # print("Using FakeBrisbane backend.\n")
    else:
        
        backend = AerSimulator(noise_model=noise_model, device="GPU",cuStateVec_enable=True)
    passmanager = generate_preset_pass_manager(
        backend=backend,
    )
    echo_str = "echo" if echo else "forward"
    backend_name = getattr(backend, 'name', str(type(backend).__name__))
    # gate_counts = circ_tnoise.count_ops()
    # gate_count_filename = f"{folder_name}/gate_counts_t{t}_{echo_str}.csv"
    
    # pd.DataFrame(list(gate_counts.items()), columns=["gate", "count"]).to_csv(gate_count_filename, index=False)
    optimization_level = 3
    circ_qubits = [circ.qubits[i] for i in range(L)]
    # snake_layout = [15,30,17,12,11,10,9,8,7,6,5,4,3,2,1,0,14,18,19,20,21]
    snake_layout = [30,17,12,11,10,9,8,7,6,5,4,3,2,1,0,14,18,19,20,21]
    layout_dict = {circ_qubits[i]: snake_layout[i] for i in range(L)}
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
    circ_tnoise = passmanager.run(circ)


    # layout_fig = plot_circuit_layout(circ_tnoise, backend=backend)
    # layout_filename = f"{folder_name}/circuit_layout_t{t}_{echo_str}_opt{optimization_level}_{backend_name}.png"
    # layout_fig.savefig(layout_filename)
    # plt.close(layout_fig)


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

def power_law_func(x, a, b, c):
    """Power law function: ax^b + c"""
    return a * np.power(x, b) + c

energies = []
# nprobs = np.arange(0, noise_prob+0.01, 0.05)
nprobs = ["fakebrisbane"]
state = initial_state

energy = get_instances(state, echo=False, noise_model=noise_model)
av_energy = np.mean(energy, axis=0)
energies.append(av_energy)
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


# if __name__ == "__main__":
plt.figure(figsize=(14, 10))

# Define contrasting colors for better visibility
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
fit_colors = ['#0d4f8c', '#cc5500', '#1a6b1a', '#a01f1f']  # Darker versions for fits

# Plot original data and fits
for i, e in enumerate(energies):
    color = colors[i % len(colors)]
    fit_color = fit_colors[i % len(fit_colors)]
    plt.plot(ts, e, 'o-', color=color, label=f'$p = {nprobs[i]}$ (data)', alpha=0.7, markersize=4)
    
    # Fit the power law function ax^b + c
    try:
        # Use only non-zero time points for fitting (avoid x=0 in power law)
        fit_ts = ts[1:]  # Skip t=0
        fit_e = e[1:]    # Skip corresponding energy value
        
        # Initial guess for parameters [a, b, c]
        initial_guess = [1.0, -0.5, np.mean(fit_e)]
        
        # Set bounds for parameters: [a_min, b_min, c_min], [a_max, b_max, c_max]
        # Constrain amplitude 'a' to reasonable values, allow negative exponent 'b', constrain 'c'
        bounds = ([-100, -3, -np.inf], [100, 3, np.inf])
        
        # Perform curve fitting with constraints
        popt, pcov = curve_fit(power_law_func, fit_ts, fit_e, p0=initial_guess, bounds=bounds, maxfev=5000)
        a_fit, b_fit, c_fit = popt
        
        # Generate smooth curve for plotting
        t_smooth = np.linspace(1, max(ts), 100)
        e_fit = power_law_func(t_smooth, a_fit, b_fit, c_fit)
        
        plt.plot(t_smooth, e_fit, '-', color=fit_color,
                label=f'$p = {nprobs[i]}$ fit: ${a_fit:.3f} \\cdot t^{{{b_fit:.3f}}} + {c_fit:.3f}$', 
                alpha=0.9, linewidth=2, markersize=6)
        
        # Print fit parameters
        print(f"Noise prob {nprobs[i]}: a={a_fit:.6f}, b={b_fit:.6f}, c={c_fit:.6f}")
        
        # Calculate R-squared
        ss_res = np.sum((fit_e - power_law_func(fit_ts, a_fit, b_fit, c_fit)) ** 2)
        ss_tot = np.sum((fit_e - np.mean(fit_e)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        print(f"R-squared for noise prob {nprobs[i]}: {r_squared:.6f}")
        
        # Add text annotation showing the power (exponent) on the plot
        # Position text close to the corresponding fit curve
        text_x = max(ts) * 0.5  # 75% along x-axis
        # Get the y-value of the fit curve at text_x position
        text_y_fit = power_law_func(text_x, a_fit, b_fit, c_fit)
        # Special positioning for different noise probabilities
        if nprobs[i] == 0.001 or nprobs[i] == 0.01:
            # Move 0.001 annotation higher up
            text_y = text_y_fit + abs(text_y_fit) * 0.15  # Higher offset for 0.001
        else:
            # Default offset for other noise probabilities
            text_y = text_y_fit - abs(text_y_fit) * 0.2  # Standard offset
        plt.text(text_x, text_y, f'$b = {b_fit:.3f}$', 
                color=fit_color, fontsize=14, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9, edgecolor=fit_color))
        
    except Exception as e:
        print(f"Fitting failed for noise prob {nprobs[i]}: {str(e)}")
        fit_color = fit_colors[i % len(fit_colors)]
        plt.plot(ts, e, '-', color=fit_color, label=f'$p = {nprobs[i]}$ (fit failed)', alpha=0.8)

plt.xlabel('Time $t$')
plt.ylabel('Energy $E$')
plt.legend(loc='upper left', framealpha=0.9)
plt.title(f'Energy for {state} state ($g={g}$, $L={L}$) with Power Law Fits')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
plot_filename = f"energy_plot_{state}_g{g}_L{L}_inst{inst}_tf{args.tf}.png"
plot_path = f"{folder_name}/{plot_filename}"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {plot_path}")

plt.show()