
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
from qiskit_ibm_runtime.fake_provider import FakeBrisbane, FakeTorino
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
from qiskit.visualization import plot_circuit_layout, plot_coupling_map
import os

from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

QiskitRuntimeService.save_account(channel='ibm_quantum_platform', token="",set_as_default=True,overwrite=True)

service = QiskitRuntimeService()


parser = argparse.ArgumentParser(description="Plot figures")
parser.add_argument("--L", type=int, default=132, help="Number of qubits")
parser.add_argument("--device_name", type=int, default=0, help="Device name")
parser.add_argument("--inst", type=int, default=2, help="Number of instances for fig3d")
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


folder_name = f"autocorr_data_L{L}_ibm_torino"

if not os.path.exists(folder_name):
    os.makedirs(folder_name)


# Load hs and phis from CSV files
disorder_folder = "."
hs_filename = os.path.join(disorder_folder, f"hs_L{L}_inst{1}_ampl{phi_amplitude}_delta{phi_delta}_randomphi{args.randomphi}.csv")
phis_filename = os.path.join(disorder_folder, f"phis_L{L}_inst{1}_ampl{phi_amplitude}_delta{phi_delta}_randomphi{args.randomphi}.csv")

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

    # circ.draw("mpl")
    # plt.show()
    # Select backend
    if args.use_fakebackend == 1:
        # backend = FakeBrisbane()
        backend = FakeTorino()
        # print("Using FakeBrisbane backend.")
    else:
        # backend = AerSimulator(noise_model=noise_model, device="GPU",cuStateVec_enable=True)
        backend = service.backend(name='ibm_torino')
    # Transpile circuit for noisy basis gates
    # Define a linear coupling map for L+1 qubits
    # coupling_list = [(i, i+1) for i in range(1,L)] + [(0, int(L/2))]
    # # print(coupling_list)
    # # print(dict(coupling_list))
    # coupling_map = CouplingMap(couplinglist=coupling_list)
    # coords = [(i, i**2/10) for i in range(L+1)]
    # fig = plot_coupling_map(L+1, coords, coupling_list)
    # coupling_map_filename = f"{folder_name}/coupling_map_L{L+1}.png"
    # fig.savefig(coupling_map_filename)
    # plt.close(fig)

    # use_coupling = len(coupling_list) > 0
    # coupling_str = "coupling" if use_coupling else "nocoupling"
    if echo:
        optimization_level = 0
    else:
        optimization_level = 3  # Use higher optimization level for forward evolution
   
    # Make initial layout follow the coupling map
    # Use the first L+1 unique nodes from the coupling map
    circ_qubits = [circ.qubits[i] for i in range(L+1)]
    # snake_layout = [15,30,17,12,11,10,9,8,7,6,5,4,3,2,1,0,14,18,19,20,21]
    snake_layout = [
        74,20,19,15,0,1,2,3,4,16,5,6,7,8,17,9,10,11,12,13,14,
        18,31,32,33,37,52,51,50,56,49,48,47,36,29,30,28,27,26,25,35,24,23,22,21,34,40,41,
        39,38,53,57,58,59,72,60,61,62,54,42,43,44,45,46,55,65,64,66,67,68,69,70,71,75,90,89,
        88,94,87,86,85,84,93,83,82,73,63,81,80,92,79,78,77,76,91,95,96,97,110,98,99,100,101,
        111,102,103,104,105,112,106,107,108,109,113,128,127,126,132,125,124,123,122,131,121,120,119,118,130,117,116,115,114,129
    ]


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
    # gate_counts = circ_tnoise.count_ops()
    # print(gate_counts)
    # Save gate counts as CSV
    # gate_count_filename = f"{folder_name}/gate_counts_t{t}_{echo_str}_opt{optimization_level}_{backend_name}-ibm_torino.csv"
    # pd.DataFrame(list(gate_counts.items()), columns=["gate", "count"]).to_csv(gate_count_filename, index=False)
    
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

    sampler = SamplerV2(mode=backend)
    result = sampler.run([circ_tnoise],shots=1024).result()
    counts = result[0].data.c.get_counts()
    expval = compute_z_expectation(counts, 1)[0]
    # print(z_expectations)
    # out = z_expectations

    # results.append(out)


    # result = backend.run(circ_tnoise).result()
    # counts = result.get_counts(circ_tnoise)
    # expval = compute_z_expectation(counts, 1)[0]
    print(expval)
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

# state = np.random.random(2**L)
# state = state / np.linalg.norm(state)  # normalize the state
# state = np.zeros(2**L)
# state[0] = 1 

state = initial_state
autocorr = get_instances(state, echo=False)
av_autocorr = np.mean(autocorr, axis=0)
# print(av_autocorr)
autocorr_echo = get_instances(state, echo=True)
av_autocorr_echo = np.mean(autocorr_echo, axis=0)

# Save averaged data
data = {
    'time': ts,
    'av_autocorr': av_autocorr,
    'av_autocorr_echo': av_autocorr_echo,
    # 'sqrt_av_autocorr_echo': np.sqrt(av_autocorr_echo)
}
df = pd.DataFrame(data)
csv_filename = f"autocorr_data_{state}_g{g}_L{L}_inst{inst}_randomphi{args.randomphi}_delta{phi_delta}_amplitude{phi_amplitude}_noise{noise_prob}_usenoise{use_noise}.csv"

csv_path = f"{folder_name}/{csv_filename}"
df.to_csv(csv_path, index=False)
print(f"Autocorrelation data saved to {csv_path}")

# Save individual instance data
# Forward autocorrelation instances
forward_instance_data = {'time': ts}
for i in range(inst):
    forward_instance_data[f'instance_{i}_forward'] = autocorr[i]

df_forward_instances = pd.DataFrame(forward_instance_data)
forward_instances_filename = f"autocorr_instances_forward_{state}_g{g}_L{L}_inst{inst}_randomphi{args.randomphi}_delta{phi_delta}_amplitude{phi_amplitude}_noise{noise_prob}_usenoise{use_noise}.csv"
forward_instances_path = f"{folder_name}/{forward_instances_filename}"
df_forward_instances.to_csv(forward_instances_path, index=False)
print(f"Forward autocorrelation instances saved to {forward_instances_path}")

# Echo autocorrelation instances
echo_instance_data = {'time': ts}
for i in range(inst):
    echo_instance_data[f'instance_{i}_echo'] = autocorr_echo[i]

df_echo_instances = pd.DataFrame(echo_instance_data)
echo_instances_filename = f"autocorr_instances_echo_{state}_g{g}_L{L}_inst{inst}_randomphi{args.randomphi}_delta{phi_delta}_amplitude{phi_amplitude}_noise{noise_prob}_usenoise{use_noise}.csv"
echo_instances_path = f"{folder_name}/{echo_instances_filename}"
df_echo_instances.to_csv(echo_instances_path, index=False)
print(f"Echo autocorrelation instances saved to {echo_instances_path}")

# print(autocorr)
# plt.plot(av_autocorr,label=f"U_F")
plt.plot(av_autocorr_echo,label=f"U_ECHO")
plt.plot(np.sqrt(av_autocorr_echo),label=f"\sqrt(U_ECHO)")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Autocorrelation")
plt.title(f"Autocorrelation for {state} state with g={g}, L={L}, δ={phi_delta}, A={phi_amplitude}, noise={'ON' if use_noise else 'OFF'}({noise_prob})")
plt.show()
