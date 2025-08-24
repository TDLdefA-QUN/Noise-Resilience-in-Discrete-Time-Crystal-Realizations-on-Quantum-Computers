
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
# from qiskit_ibm_runtime.fake_provider import FakeBrisbane
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
from qiskit.visualization import plot_circuit_layout, plot_coupling_map
import os

from iqm.qiskit_iqm import IQMCircuit, IQMProvider, transpile_to_IQM
from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet

parser = argparse.ArgumentParser(description="Plot figures")
parser.add_argument("--L", type=int, default=19, help="Number of qubits")
parser.add_argument("--shots", type=int, default=10, help="Number of shots for circuit execution")
parser.add_argument("--device_name", type=int, default=0, help="Device name")
parser.add_argument("--inst", type=int, default=1, help="Number of instances for fig3d")
parser.add_argument("--randomphi", type=int, default=1, help="Prethermal=0 or DTC=1")
parser.add_argument("--phi_delta", type=float, default=0.0, help="Phi delta parameter")
parser.add_argument("--phi_amplitude", type=float, default=1.0, help="Phi amplitude parameter")
parser.add_argument("--tf",type=int, default=5, help="end time for fig3d")
parser.add_argument("--g",type=float, default=0.95, help="g for fig3d")
parser.add_argument("--noise_prob",type=float, default=0.05, help="noise probability")
parser.add_argument("--use_noise",type=int, default=1, help="use depolarizing noise: 0=no noise, 1=apply noise")
parser.add_argument("--initial_state",type=str, default="vacuum", help="initial state")
parser.add_argument("--use_fakebackend", type=int, default=1, help="Use FakeBackend for simulation: 0=No, 1=Yes")

args = parser.parse_args()
shots = args.shots

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


folder_name = f"data_L{L}_iqm"

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

def create_UF_subcircuit(L, g, phis, hs):
    subcircuit = QuantumCircuit(L+1)
    for i in range(L):
        subcircuit.rx(np.pi*g, i+1)
    for i in range(0, L-1, 2):
        subcircuit.rzz(phis[i], i+1, i+2)
    for i in range(1, L-1, 2):
        subcircuit.rzz(phis[i], i+1, i+2)
    for i in range(L):
        subcircuit.rz(hs[i], i+1)
    return subcircuit


def qc_qiskit(initial_state, L, g, hs, phis, t, qubit, echo=False):
    echo_str = "echo" if echo else "forward"
    coupling_list = [(i, i+1) for i in range(1,L)] + [(0, int(L/2))]
    
    coupling_map = CouplingMap(couplinglist=coupling_list)
    coords = [(i, i**2/10) for i in range(L+1)]

    use_coupling = len(coupling_list) > 0
    coupling_str = "coupling" if use_coupling else "nocoupling"
    optimization_level = 0
    routing_method = "lookahead"  # You can change to "lookahead", "sabre", etc.
    layout_method = "dense"
    circ = QuantumCircuit(L+1,1)
    # Initial state
    if initial_state == "neel":
        for i in range(1, L+1):
            if i % 2 == 0:
                circ.x(i)
    # Hadamard and entangle
    circ.h(0)
    circ.cz(qubit+1, 0)
    # Pre-build UF subcircuit
    UF_subcircuit = create_UF_subcircuit(L, g, phis, hs)
    # Forward evolution
    for _ in range(t):
        circ.append(UF_subcircuit, range(L+1))
    # Echo evolution
    if echo:
        UF_inv = UF_subcircuit.inverse()
        for _ in range(t):
            circ.append(UF_inv, range(L+1))

    circ.cz(qubit+1, 0)
    circ.h(0)
    circ.measure(0,0)

    # Select backend
    if args.use_fakebackend == 1:
        backend = IQMFakeGarnet()
        print("Using IQMFakeGarnet backend.")
    else:
        # backend = AerSimulator(noise_model=noise_model, device="GPU",cuStateVec_enable=True)
        server_url = "https://cocos.resonance.meetiqm.com/garnet"
        api_token = ""
        backend = IQMProvider(server_url, token=api_token).get_backend()
    backend_name = getattr(backend, 'name', str(type(backend).__name__))

    circ_drawing = circ.draw("mpl")
    circ_drawing.savefig(f"{folder_name}/initial_circuit_t{t}_{echo_str}_opt{optimization_level}_{backend_name}_{coupling_str}_route{routing_method}_layout{layout_method}_iqm.png",dpi=300)
    plt.close(circ_drawing)

    fig = plot_coupling_map(L+1, coords, coupling_list)
    coupling_map_filename = f"{folder_name}/coupling_map_L{L+1}_iqm.png"
    fig.savefig(coupling_map_filename,dpi=300)
    plt.close(fig)

    
    # Make initial layout follow the coupling map
    # Use the first L+1 unique nodes from the coupling map
    initial_layout = Layout(
        {
            circ.qubits[0]: 14,
            circ.qubits[1]: 0,
            circ.qubits[2]: 1,
            circ.qubits[3]: 4,
            circ.qubits[4]: 5,
            circ.qubits[5]: 6,
            circ.qubits[6]: 11,
            circ.qubits[7]: 16,
            circ.qubits[8]: 15,
            circ.qubits[9]: 19,
            circ.qubits[10]: 18,
            circ.qubits[11]: 17,
            circ.qubits[12]: 13,
            circ.qubits[13]: 12,
            circ.qubits[14]: 7,
            circ.qubits[15]: 2,
            circ.qubits[16]: 3,
            circ.qubits[17]: 8,
            circ.qubits[18]: 9,
            circ.qubits[19]: 10,
        }
    )
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
    gate_counts = circ_tnoise.count_ops()
    # print(gate_counts)
    # Save gate counts as CSV
    gate_count_filename = f"{folder_name}/gate_counts_t{t}_{echo_str}_opt{optimization_level}_{backend_name}_{coupling_str}_route{routing_method}_layout{layout_method}_iqm.csv"
    pd.DataFrame(list(gate_counts.items()), columns=["gate", "count"]).to_csv(gate_count_filename, index=False)

    layout_fig = plot_circuit_layout(circ_tnoise, backend=backend)
    # layout_str = "-".join(str(q) for q in initial_layout)
    layout_filename = f"{folder_name}/circuit_layout_t{t}_{echo_str}_opt{optimization_level}_{backend_name}_{coupling_str}_route{routing_method}_layout{layout_method}_iqm.png"
    layout_fig.savefig(layout_filename,dpi=300)
    plt.close(layout_fig)

    # Draw and save the transpiled circuit
    circuit_fig = circ_tnoise.draw(output='mpl')
    circuit_filename = f"{folder_name}/transpiled_circuit_t{t}_{echo_str}_opt{optimization_level}_{backend_name}_{coupling_str}_route{routing_method}_layout{layout_method}_iqm.png"
    circuit_fig.savefig(circuit_filename,dpi=300)
    plt.close(circuit_fig)
    # Run and get counts
    result = backend.run(circ_tnoise, shots=shots).result()
    counts = result.get_counts(circ_tnoise)
    expval = compute_z_expectation(counts, 1)[0]
    return expval


def get_single_out(initial_state, inst_number, echo):
    results = []
    for t in range(T):
        # print(f"time: {t}")
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

# state = np.random.random(2**L)
# state = state / np.linalg.norm(state)  # normalize the state
# state = np.zeros(2**L)
# state[0] = 1 

state = initial_state
autocorr = get_instances(state, echo=False)
av_autocorr = np.mean(autocorr, axis=0)
# print(av_autocorr)
# autocorr_echo = get_instances(state, echo=True)
# av_autocorr_echo = np.mean(autocorr_echo, axis=0)

data = {
    'time': ts,
    'av_autocorr': av_autocorr,
    # 'av_autocorr_echo': av_autocorr_echo,
    # 'sqrt_av_autocorr_echo': np.sqrt(av_autocorr_echo)
}
df = pd.DataFrame(data)
csv_filename = f"autocorr_data_{state}_g{g}_L{L}_inst{inst}_randomphi{args.randomphi}_delta{phi_delta}_amplitude{phi_amplitude}_noise{noise_prob}_usenoise{use_noise}_iqm.csv"

csv_path = f"{folder_name}/{csv_filename}"
df.to_csv(csv_path, index=False)
print(f"Autocorrelation data saved to {csv_path}")

# print(autocorr)
#plt.plot(av_autocorr,label=f"U_F")
#plt.plot(av_autocorr_echo,label=f"U_ECHO")
#plt.plot(np.sqrt(av_autocorr_echo),label=f"\sqrt(U_ECHO)")
#plt.legend()
#plt.xlabel("Time (s)")
#plt.ylabel("Autocorrelation")
#plt.title(f"Autocorrelation for {state} state with g={g}, L={L}, δ={phi_delta}, A={phi_amplitude}, noise={'ON' if use_noise else 'OFF'}({noise_prob})")
#plt.show()
