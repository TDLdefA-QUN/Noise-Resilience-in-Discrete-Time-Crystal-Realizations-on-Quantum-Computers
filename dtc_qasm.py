import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
import time
import argparse
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2

parser = argparse.ArgumentParser(description="Plot figures")
parser.add_argument("--L", type=int, default=10, help="Number of qubits")
parser.add_argument("--device_name", type=int, default=0, help="Device name")
parser.add_argument("--inst", type=int, default=20, help="Number of instances for fig3d")
parser.add_argument("--randomphi", type=int, default=1, help="Prethermal=0 or DTC=1")
parser.add_argument("--phi_delta", type=float, default=0.0, help="Prethermal=0 or DTC=1")
parser.add_argument("--phi_amplitude", type=float, default=1.0, help="Prethermal=0 or DTC=1")
parser.add_argument("--tf",type=int, default=30, help="end time for fig3d")
parser.add_argument("--g",type=float, default=0.94, help="g for fig3d")
parser.add_argument("--mpi",type=int, default=0, help="using mpi")
parser.add_argument("--nodes",type=int, default=1, help="number of nodes")
args = parser.parse_args()

L = args.L
g = args.g
inst = args.inst
use_mpi = args.mpi
phi_delta = args.phi_delta
phi_amplitude = args.phi_amplitude

t_start = 0
t_end = args.tf
T = t_end - t_start
ts = np.arange(t_start, t_end, 1)

hs = np.random.random((inst,L)) * 2*np.pi - np.pi # [-pi, pi]
if args.randomphi == 1:
    phis = np.random.random((inst,L-1)) * phi_amplitude *np.pi - 1.5 * np.pi + phi_delta*np.pi # [-1.5 pi, -0.5pi]
else:
    phis = np.full((inst,L-1), -0.4)


dn = args.device_name

if dn == 0:
    device_name = "qubit"
elif dn == 1:
    device_name = "gpu"
elif dn == 2:
    device_name = "tensor"
elif dn == 3:
    device_name = "kokkos"
    

if use_mpi:
    dev = qml.device(f"lightning.{device_name}", wires=L, shots=None, mpi=True, batch_obs=True) 
else:

    dev = qml.device(f"lightning.{device_name}", wires=L, shots=None)

# backend = AerSimulator()

QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token="",set_as_default=True, overwrite=True)
service = QiskitRuntimeService()
backend = service.backend(name="ibm_brisbane")


@qml.qnode(dev,diff_method=None)
def qc(state, L, g, hs, phis, t):
    #print(f"hi from qc rank = {rank}")
    if state == "0":
        pass
    elif state == "1":
        qml.X(wires=int(L/2))
    
    for t_ in range(t):
        for i in range(L):
            qml.RX(np.pi*g, wires=i)
        
        for i in range(0,L-1,2):
            qml.IsingZZ(phis[i], wires=[i, i+1])

        for i in range(1,L-1,2):
            qml.IsingZZ(phis[i], wires=[i, i+1])

        for i in range(L):
            qml.RZ(hs[i], wires=i)

    return [qml.expval(qml.PauliZ(i)) for i in range(L)]



qasm_qc = qml.to_openqasm(qc)


def save_qasm(qasm_code, filename):
    with open(filename, 'w') as f:
        f.write(qasm_code)

def qasm_to_qiskit(qasm_file):
    with open(qasm_file, 'r') as f:
        qasm_code = f.read()

    qc = QuantumCircuit.from_qasm_str(qasm_code)
    return qc

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

def get_single_out(initial_state,inst_number):
    this_state = initial_state
    results = []
    for t in range(1,T):

        # out = qc(this_state, L, g, hs[inst_number], phis[inst_number], t)
        qasm_code = qasm_qc(this_state, L, g, hs[inst_number], phis[inst_number], t)
        save_qasm(qasm_code, f"qasm_output_{inst_number}_t{t}.qasm")
        qiskit_qc = qasm_to_qiskit(f"qasm_output_{inst_number}_t{t}.qasm")
        # qiskit_qc.draw("mpl")
        # plt.show()
        pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
        compiled_qc = pm.run(qiskit_qc)
        
        sampler = SamplerV2(mode=backend)
        result = sampler.run([compiled_qc],shots=1024).result()
        counts = result[0].data.c.get_counts()
        z_expectations = compute_z_expectation(counts, L)
        # print(z_expectations)
        out = z_expectations

        results.append(out)
        # with Session(backend=backend) as session:
        #     sampler = SamplerV2(mode=session)
        #     result = sampler.run([compiled_qc],shots=1024).result()
        #     counts = result[0].data.c.get_counts()
        #     z_expectations = compute_z_expectation(counts, L)
        #     # print(z_expectations)
        #     out = z_expectations

        #     results.append(out)
        
    results = np.array(results)
    return results.T


def get_instances(initial_state):
    all_results = []
    for i in range(inst):
        results = get_single_out(initial_state,i)
        all_results.append(results)
    all_results = np.array(all_results)
    return all_results

def savecsv(array, name):
    m,n,r = array.shape
    arr = np.column_stack((np.repeat(np.arange(m),n),array.reshape(m*n,-1)))
    df = pd.DataFrame(arr)
    df.to_csv(name)

start = time.time()
instances = get_instances("0")
end = time.time()
print(f"Time taken: {end - start} seconds")

savecsv(instances,f"expz-nodes{args.nodes}-{L}-{g}-{inst}-{t_end}-{args.randomphi}-{phi_delta}-{phi_amplitude}-qiskit-{use_mpi}.csv")

av = np.mean(instances, axis=0)
plt.plot(av[int(L/2)])
plt.show()
# print(instances)

