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
from qiskit.transpiler import generate_preset_pass_manager
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

parser = argparse.ArgumentParser(description="Plot figures")
parser.add_argument("--L", type=int, default=20, help="Number of qubits")
parser.add_argument("--device_name", type=int, default=0, help="Device name")
parser.add_argument("--inst", type=int, default=30, help="Number of instances for fig3d")
parser.add_argument("--randomphi", type=int, default=0, help="Prethermal=0 or DTC=1")
parser.add_argument("--phi_delta", type=float, default=0.0, help="Phi delta parameter")
parser.add_argument("--phi_amplitude", type=float, default=1.0, help="Phi amplitude parameter")
parser.add_argument("--tf",type=int, default=30, help="end time for fig3d")
parser.add_argument("--g",type=float, default=0.94, help="g for fig3d")
parser.add_argument("--noise_prob",type=float, default=0.05, help="noise probability")
parser.add_argument("--use_noise",type=int, default=1, help="use depolarizing noise: 0=no noise, 1=apply noise")
parser.add_argument("--initial_state",type=str, default="vacuum", help="initial state")

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

# Generate random parameters
hs = np.random.random((inst,L)) * 2*np.pi - np.pi # [-pi, pi]
if args.randomphi == 1:
    phis = np.random.random((inst,L-1)) * phi_amplitude * np.pi - 1.5 * np.pi + phi_delta * np.pi # [-1.5 pi, -0.5pi]
else:
    phis = np.full((inst,L-1), -0.4)

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

def UF(circ, L, g, phis, hs):
    for i in range(L):
        circ.rx(np.pi*g, i+1)
    for i in range(0, L-1, 2):
        circ.rzz(phis[i], i+1, i+2)
    for i in range(1, L-1, 2):
        circ.rzz(phis[i], i+1, i+2)
    for i in range(L):
        circ.rz(hs[i], i+1)
    return circ

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
    # Forward evolution
    for _ in range(t):
        UF(circ, L, g, phis, hs)
    # Echo evolution
    if echo:
        for _ in range(t):
            UF(circ, L, g, phis, hs).inverse()

    circ.cz(qubit+1, 0)
    circ.h(0)
    circ.measure(0,0)

    sim_noise = AerSimulator(noise_model=noise_model, device="GPU",cuStateVec_enable=True)
    # Transpile circuit for noisy basis gates
    passmanager = generate_preset_pass_manager(
        optimization_level=0, backend=sim_noise
    )
    circ_tnoise = passmanager.run(circ)
    
    # Run and get counts
    result = sim_noise.run(circ_tnoise).result()
    counts = result.get_counts(circ_tnoise)
    # print(counts)
    expval = compute_z_expectation(counts, 1)[0]
    # print(expval)
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
print(av_autocorr)
autocorr_echo = get_instances(state, echo=True)
av_autocorr_echo = np.mean(autocorr_echo, axis=0)

data = {
    'time': ts,
    'av_autocorr': av_autocorr,
    'av_autocorr_echo': av_autocorr_echo,
    'sqrt_av_autocorr_echo': np.sqrt(av_autocorr_echo)
}
df = pd.DataFrame(data)
csv_filename = f"autocorr_data_{state}_g{g}_L{L}_inst{inst}_randomphi{args.randomphi}_delta{phi_delta}_amplitude{phi_amplitude}_noise{noise_prob}_usenoise{use_noise}.csv"
folder_name = f"data_L{L}"
import os
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"Created folder: {folder_name}")
else:
    print(f"Folder already exists: {folder_name}")
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
