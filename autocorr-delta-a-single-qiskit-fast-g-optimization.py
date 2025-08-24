import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import pandas as pd
from functools import partial
from scipy.optimize import curve_fit, minimize_scalar
from scipy.signal import find_peaks
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

color_palette = ["#361AC1", "#15B300", "#E33100", "#DF349E", "#0C8BCA", "#FF9100", "#E72142", '#AA4499']

def find_envelope(signal, window_size=5):
    """Find the upper and lower envelope of a signal using rolling max/min with interpolation for smoothness"""
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


parser = argparse.ArgumentParser(description="Plot figures")
parser.add_argument("--L", type=int, default=20, help="Number of qubits")
parser.add_argument("--device_name", type=int, default=0, help="Device name")
parser.add_argument("--inst", type=int, default=1, help="Number of instances for fig3d")
parser.add_argument("--randomphi", type=int, default=1, help="Prethermal=0 or DTC=1")
parser.add_argument("--phi_delta", type=float, default=0.0, help="Phi delta parameter")
parser.add_argument("--phi_amplitude", type=float, default=1.0, help="Phi amplitude parameter")
parser.add_argument("--tf",type=int, default=20, help="end time for fig3d")
parser.add_argument("--g",type=float, default=0.84, help="g for fig3d")
parser.add_argument("--noise_prob",type=float, default=0.05, help="noise probability")
parser.add_argument("--use_noise",type=int, default=1, help="use depolarizing noise: 0=no noise, 1=apply noise")
parser.add_argument("--initial_state",type=str, default="vacuum", help="initial state")
parser.add_argument("--use_fakebackend", type=int, default=0, help="Use FakeBackend for simulation: 0=No, 1=Yes")
parser.add_argument("--target_echo", type=float, default=1.0, help="Target echo value for feedback control")
parser.add_argument("--feedback_gain", type=float, default=0.01, help="Feedback gain for g adjustment")
parser.add_argument("--exponential_feedback", type=int, default=1, help="Use exponential feedback: 0=linear, 1=exponential")
parser.add_argument("--decay_compensation", type=float, default=0.1, help="Exponential decay compensation factor")
parser.add_argument("--g_min", type=float, default=0.84, help="Minimum allowed g value")
parser.add_argument("--g_max", type=float, default=1.0, help="Maximum allowed g value")
parser.add_argument("--use_optimization", type=int, default=1, help="Use optimization to minimize echo distance: 0=feedback control, 1=optimization")
parser.add_argument("--optimization_iterations", type=int, default=5 , help="Number of optimization iterations per time step")

args = parser.parse_args()

L = args.L
g_initial = args.g  # Store initial g value
inst = args.inst
phi_delta = args.phi_delta
phi_amplitude = args.phi_amplitude
use_noise = args.use_noise
target_echo = args.target_echo
feedback_gain = args.feedback_gain
exponential_feedback = args.exponential_feedback
decay_compensation = args.decay_compensation
g_min = args.g_min
g_max = args.g_max
use_optimization = args.use_optimization
optimization_iterations = args.optimization_iterations

t_start = 0
t_end = args.tf
T = t_end - t_start
ts = np.arange(t_start, t_end, 1)
noise_prob = args.noise_prob
initial_state = args.initial_state


folder_name = f"controlled-autocorr_data_L{L}"

if not os.path.exists(folder_name):
    os.makedirs(folder_name)


# Load hs and phis from CSV files
disorder_folder = "disorder_data"
hs_filename = os.path.join(disorder_folder, f"hs_L{L}_inst{100}_ampl{phi_amplitude}_delta{phi_delta}_randomphi{args.randomphi}.csv")
phis_filename = os.path.join(disorder_folder, f"phis_L{L}_inst{100}_ampl{phi_amplitude}_delta{phi_delta}_randomphi{args.randomphi}.csv")
# disorder_folder = "."
# hs_filename = os.path.join(disorder_folder, f"hs_L{L}.csv")
# phis_filename = os.path.join(disorder_folder, f"phis_L{L}.csv")

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


def qc_qiskit(initial_state, L, g_values, hs, phis, t, qubit, echo=False):
    """
    Modified to accept g_values as a list where g_values[i] is the g value to use at time step i
    For time t, we use g_values[0] for the first UF, g_values[1] for the second UF, etc.
    """
    circ = QuantumCircuit(L+1,1)
    # Initial state
    if initial_state == "neel":
        for i in range(1, L+1):
            if i % 2 == 0:
                circ.x(i)
    # Hadamard and entangle
    circ.h(0)
    circ.cz(qubit+1, 0)
    
    # Forward evolution - use different g values for each time step
    for time_step in range(t):
        # Determine which g value to use for this time step
        if isinstance(g_values, list) and len(g_values) > time_step:
            current_g = g_values[time_step]
        elif isinstance(g_values, (int, float)):
            current_g = g_values  # Backward compatibility
        else:
            current_g = g_values[0] if len(g_values) > 0 else 0.84  # Fallback
            
        UF_subcircuit = create_UF_subcircuit(L, current_g, phis, hs)
        circ.append(UF_subcircuit, range(L+1))
    
    # Echo evolution - apply inverse in reverse order
    if echo:
        for time_step in range(t-1, -1, -1):  # Reverse order for echo
            # Use the same g value that was used in forward evolution
            if isinstance(g_values, list) and len(g_values) > time_step:
                current_g = g_values[time_step]
            elif isinstance(g_values, (int, float)):
                current_g = g_values
            else:
                current_g = g_values[0] if len(g_values) > 0 else 0.84
                
            UF_subcircuit = create_UF_subcircuit(L, current_g, phis, hs)
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
    gate_count_filename = f"{folder_name}/gate_counts_t{t}_{echo_str}_opt{optimization_level}_{backend_name}_{coupling_str}_route{routing_method}_layout{layout_method}_iqm.csv"
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


def get_single_out_adaptive(initial_state, inst_number, echo, g_values=None):
    """
    Modified version that uses time-dependent g values for forward evolution
    and computes echo to provide feedback for g adjustment
    """
    results = []
    echo_results = []
    current_g_values = g_values if g_values is not None else [g_initial] * T
    
    for t in range(T):
        print(f"time: {t}, using g_values: [{', '.join([f'{g:.3f}' for g in current_g_values[:t+1]])}]")
        
        # Use g values up to time t for the circuit
        g_values_for_circuit = current_g_values[:t+1]
        
        if echo:
            # For echo simulation, use the g values up to this time step
            out = qc_qiskit(initial_state, L, g_values_for_circuit, hs[inst_number], phis[inst_number], t+1, int(L/2), echo=True)
            echo_results.append(out)
        else:
            # For forward simulation, use the g values up to this time step
            out = qc_qiskit(initial_state, L, g_values_for_circuit, hs[inst_number], phis[inst_number], t+1, int(L/2), echo=False)
        
        results.append(out)
    
    results = np.array(results)
    if echo:
        return results.T, np.array(echo_results)
    return results.T

def adjust_g_based_on_echo(echo_values, current_g_values, target_echo, feedback_gain, g_min, g_max):
    """
    Adjust g values based on echo feedback to keep echo close to target value
    """
    new_g_values = current_g_values.copy()
    
    for t in range(len(echo_values)):
        if t > 0:  # Start adjusting from second time step
            echo_error = target_echo - echo_values[t-1]  # Use previous echo for feedback
            g_adjustment = feedback_gain * echo_error
            new_g_values[t] = np.clip(current_g_values[t] + g_adjustment, g_min, g_max)
    
    return new_g_values

def optimize_g_for_target_echo(initial_state, inst_number, current_time, g_values_history, target_echo, g_min, g_max, num_iterations=5):
    """
    Optimize g value by minimizing the distance between echo result and target_echo
    
    Args:
        initial_state: Initial quantum state
        inst_number: Instance number for disorder realization
        current_time: Current time step
        g_values_history: List of g values used up to current_time-1
        target_echo: Target echo value to achieve
        g_min, g_max: g value bounds
        num_iterations: Number of optimization iterations
    
    Returns:
        optimal_g: Optimized g value that minimizes |echo - target_echo|
    """
    from scipy.optimize import minimize_scalar
    
    def objective_function(g_candidate):
        """Objective function: squared distance between echo and target"""
        # Create g_values list for circuit up to current time
        test_g_values = g_values_history + [g_candidate]
        
        # Run echo simulation with candidate g value
        try:
            echo_val = qc_qiskit(initial_state, L, test_g_values, hs[inst_number], phis[inst_number], 
                                current_time + 1, int(L/2), echo=True)
            # Return squared distance to target
            return (echo_val - target_echo)**2
        except Exception as e:
            print(f"Error in objective function evaluation: {e}")
            return float('inf')  # Return large value if simulation fails
    
    # Use scipy's minimize_scalar with bounds
    result = minimize_scalar(objective_function, bounds=(g_min, g_max), method='bounded')
    
    if result.success:
        optimal_g = result.x
        optimal_echo_distance = np.sqrt(result.fun)
        print(f"         Optimization successful: g={optimal_g:.4f}, echo_distance={optimal_echo_distance:.4f}")
        return optimal_g
    else:
        print(f"         Optimization failed, using fallback method")
        # Fallback: grid search with fewer points
        return grid_search_g_optimization(initial_state, inst_number, current_time, 
                                         g_values_history, target_echo, g_min, g_max)

def grid_search_g_optimization(initial_state, inst_number, current_time, g_values_history, target_echo, g_min, g_max, num_points=10):
    """
    Grid search optimization as fallback method
    """
    g_candidates = np.linspace(g_min, g_max, num_points)
    best_g = g_min
    best_distance = float('inf')
    
    for g_candidate in g_candidates:
        test_g_values = g_values_history + [g_candidate]
        try:
            echo_val = qc_qiskit(initial_state, L, test_g_values, hs[inst_number], phis[inst_number], 
                                current_time + 1, int(L/2), echo=True)
            distance = abs(echo_val - target_echo)
            if distance < best_distance:
                best_distance = distance
                best_g = g_candidate
        except Exception as e:
            continue  # Skip failed evaluations
    
    print(f"         Grid search: g={best_g:.4f}, echo_distance={best_distance:.4f}")
    return best_g

def calculate_exponential_g_adjustment(echo_val, target_echo, current_g, time_step, feedback_gain, decay_compensation, g_min, g_max):
    """
    Calculate exponential adjustment for g based on echo decay
    
    Args:
        echo_val: Current echo measurement
        target_echo: Target echo value (usually 1.0)
        current_g: Current g value
        time_step: Current time step
        feedback_gain: Base feedback gain
        decay_compensation: Exponential decay compensation factor
        g_min, g_max: g value bounds
    
    Returns:
        new_g: Adjusted g value
    """
    # Calculate echo error
    echo_error = target_echo - echo_val
    
    # If echo is decaying exponentially, we expect echo ≈ target * exp(-decay_rate * t)
    # To compensate, we can increase g exponentially: g ≈ g_initial * exp(compensation * t)
    
    if exponential_feedback:
        # Method 1: Exponential compensation based on time
        time_factor = np.exp(decay_compensation * time_step)
        exponential_adjustment = feedback_gain * echo_error * time_factor
        
        # Method 2: Logarithmic error amplification (for small echo values)
        if echo_val > 0.01:  # Avoid log of very small numbers
            log_ratio = np.log(target_echo / echo_val) if echo_val < target_echo else 0
            log_adjustment = feedback_gain * log_ratio * 0.1  # Scale down log adjustment
        else:
            log_adjustment = feedback_gain * 2.0  # Strong correction for very small echo
        
        # Combine both methods
        total_adjustment = exponential_adjustment + log_adjustment
        
        # Apply exponential scaling to the adjustment itself
        scaled_adjustment = total_adjustment * (1 + decay_compensation * time_step)
        
        new_g = current_g + scaled_adjustment
    else:
        # Linear feedback (original method)
        new_g = current_g + feedback_gain * echo_error
    
    # Ensure g stays within bounds
    return np.clip(new_g, g_min, g_max)

def get_single_out(initial_state, inst_number, echo, fixed_g=None):
    """
    Standard simulation with fixed g value
    
    Args:
        fixed_g: If provided, use this fixed g value instead of g_initial
    """
    g_value = fixed_g if fixed_g is not None else g_initial
    results = []
    for t in range(T):
        print(f"time: {t}, fixed g: {g_value}")
        # For standard simulation, use the same g value for all time steps
        g_values_list = [g_value] * (t + 1)  # Use fixed g for all time steps up to t
        out = qc_qiskit(initial_state, L, g_values_list, hs[inst_number], phis[inst_number], t+1, int(L/2), echo=echo)
        results.append(out)
    results = np.array(results)
    return results.T



def get_instances_adaptive_realtime(initial_state):
    """
    Real-time adaptive simulation that adjusts g based on previous echo feedback
    Uses causality: g(t) depends only on echo measurements from t-1 and earlier
    At time t, circuit uses [g_0, g_1, ..., g_{t-1}] where g_i was determined based on echo at time i
    Supports both feedback control and optimization-based approaches
    """
    method_type = "optimization" if use_optimization else ("exponential feedback" if exponential_feedback else "linear feedback")
    print(f"\nRunning real-time adaptive simulation with {method_type} control...")
    print(f"Target echo: {target_echo}, {'Optimization iterations' if use_optimization else 'Feedback gain'}: {optimization_iterations if use_optimization else feedback_gain}")
    if not use_optimization and exponential_feedback:
        print(f"Exponential feedback enabled, decay compensation: {decay_compensation}")
    print(f"g range: [{g_min}, {g_max}], Initial g: {g_initial}")
    print("="*70)
    
    start_time = time.time()
    all_results_forward = []
    all_results_echo = []
    all_g_values = []
    
    for i in range(inst):
        print(f"\nInstance {i+1}/{inst} (real-time {method_type} control)")
        print("-" * 40)
        
        # Initialize arrays for this instance
        forward_results = []
        echo_results = []
        g_history = []  # This will store the g value used at each time step
        
        # Initialize g for first time step
        current_g = g_initial
        
        # Time evolution with real-time feedback or optimization
        for t in range(T):
            print(f"Time {t:2d}: g = {current_g:.4f}", end="")
            
            # Build g_values list for circuit up to time t
            # For time t, we need g values for time steps [0, 1, ..., t-1]
            g_values_for_circuit = g_history + [current_g]  # Include current g for this time step
            g_history.append(current_g)  # Store for future reference
            
            print(f", g_history: [{', '.join([f'{g:.3f}' for g in g_values_for_circuit])}]")
            
            # Run forward evolution at time t with accumulated g values
            forward_val = qc_qiskit(initial_state, L, g_values_for_circuit, hs[i], phis[i], t+1, int(L/2), echo=False)
            forward_results.append(forward_val)
            
            # Run echo evolution at time t with accumulated g values  
            echo_val = qc_qiskit(initial_state, L, g_values_for_circuit, hs[i], phis[i], t+1, int(L/2), echo=True)
            echo_results.append(echo_val)
            
            print(f"         → forward: {forward_val:.4f}, echo: {echo_val:.4f}")
            
            # Update g for next time step based on current echo measurement
            if t < T - 1:  # Don't update g for the last time step
                if use_optimization:
                    # Use optimization to minimize distance between echo and target
                    current_g = optimize_g_for_target_echo(
                        initial_state, i, t, g_history[:-1], target_echo, g_min, g_max, optimization_iterations
                    )
                    echo_error = target_echo - echo_val
                    print(f"         Echo error: {echo_error:.4f}, optimized next g: {current_g:.4f}")
                else:
                    # Use exponential or linear feedback control
                    current_g = calculate_exponential_g_adjustment(
                        echo_val, target_echo, current_g, t, 
                        feedback_gain, decay_compensation, g_min, g_max
                    )
                    
                    echo_error = target_echo - echo_val
                    if exponential_feedback:
                        # Show exponential feedback details
                        time_factor = np.exp(decay_compensation * t)
                        print(f"         Echo error: {echo_error:.4f}, time_factor: {time_factor:.3f}, next g: {current_g:.4f}")
                    else:
                        print(f"         Echo error: {echo_error:.4f}, linear adjustment, next g: {current_g:.4f}")
        
        # Store results for this instance
        all_results_forward.append(np.array(forward_results))
        all_results_echo.append(np.array(echo_results))
        all_g_values.append(np.array(g_history))
        
        # Print instance summary
        avg_echo = np.mean(echo_results)
        avg_g = np.mean(g_history)
        final_echo = echo_results[-1]
        final_g = g_history[-1]
        echo_distance_avg = np.mean(np.abs(np.array(echo_results) - target_echo))
        echo_distance_final = abs(final_echo - target_echo)
        
        print(f"\nInstance {i+1} Summary:")
        print(f"  Average echo: {avg_echo:.4f} (target: {target_echo:.4f})")
        print(f"  Final echo: {final_echo:.4f}")
        print(f"  Average echo distance: {echo_distance_avg:.4f}")
        print(f"  Final echo distance: {echo_distance_final:.4f}")
        print(f"  Average g: {avg_g:.4f}")
        print(f"  Final g: {final_g:.4f}")
        print(f"  g range used: [{np.min(g_history):.4f}, {np.max(g_history):.4f}]")
        
        # Print complete g value history for this instance
        print(f"  Complete g history: {', '.join([f'{g:.3f}' for g in g_history])}")
    
    elapsed = time.time() - start_time
    print(f"\nCompleted real-time {method_type} simulation in {elapsed:.2f}s")
    
    # Convert to numpy arrays
    all_results_forward = np.array(all_results_forward)
    all_results_echo = np.array(all_results_echo)
    all_g_values = np.array(all_g_values)
    
    # Print overall statistics
    print("\n" + "="*60)
    print("OVERALL STATISTICS")
    print("="*60)
    avg_g_across_all = np.mean(all_g_values)
    avg_echo_across_all = np.mean(all_results_echo)
    avg_echo_distance = np.mean(np.abs(all_results_echo - target_echo))
    final_echo_distance = np.mean(np.abs(all_results_echo[:, -1] - target_echo))
    
    print(f"Average g across all instances and times: {avg_g_across_all:.4f}")
    print(f"Average echo across all instances and times: {avg_echo_across_all:.4f}")
    print(f"Average echo distance from target: {avg_echo_distance:.4f}")
    print(f"Final time average g: {np.mean(all_g_values[:, -1]):.4f}")
    print(f"Final time average echo: {np.mean(all_results_echo[:, -1]):.4f}")
    print(f"Final time average echo distance: {final_echo_distance:.4f}")
    
    return all_results_forward, all_results_echo, all_g_values

def get_instances_adaptive(initial_state):
    """
    Adaptive simulation that adjusts g based on echo feedback
    """
    print(f"\nRunning adaptive simulation with feedback control...")
    start_time = time.time()
    all_results_forward = []
    all_results_echo = []
    all_g_values = []
    
    for i in range(inst):
        print(f"\nInstance {i+1}/{inst} (adaptive control)")
        
        # Initialize g values for this instance
        g_values = [g_initial] * T
        
        # Run initial forward and echo simulations
        results_forward = get_single_out_adaptive(initial_state, i, echo=False, g_values=g_values)
        results_echo, echo_values = get_single_out_adaptive(initial_state, i, echo=True, g_values=g_values)
        
        # Adjust g values based on echo feedback
        adjusted_g_values = adjust_g_based_on_echo(echo_values, g_values, target_echo, feedback_gain, g_min, g_max)
        
        # Run final forward simulation with adjusted g values
        print(f"Re-running forward simulation with adjusted g values...")
        results_forward_adjusted = get_single_out_adaptive(initial_state, i, echo=False, g_values=adjusted_g_values)
        
        all_results_forward.append(results_forward_adjusted)
        all_results_echo.append(results_echo)
        all_g_values.append(adjusted_g_values)
        
        # Print some statistics for this instance
        avg_echo = np.mean(echo_values)
        avg_g = np.mean(adjusted_g_values)
        print(f"Instance {i+1}: avg_echo={avg_echo:.4f}, avg_g={avg_g:.4f}")
        print(f"g history: {', '.join([f'{g:.3f}' for g in adjusted_g_values])}")
    
    elapsed = time.time() - start_time
    print(f"\nCompleted adaptive simulation in {elapsed:.2f}s")
    
    all_results_forward = np.array(all_results_forward)
    all_results_echo = np.array(all_results_echo)
    all_g_values = np.array(all_g_values)
    
    return all_results_forward, all_results_echo, all_g_values

def get_instances(initial_state, echo, fixed_g=None):
    """
    Standard simulation with fixed g value
    
    Args:
        fixed_g: If provided, use this fixed g value instead of g_initial
    """
    g_value = fixed_g if fixed_g is not None else g_initial
    print(f"\nRunning {'echo' if echo else 'forward'} simulation with fixed g={g_value:.3f} (single process)...")
    start_time = time.time()
    all_results = []
    for i in range(inst):
        print(f"Instance {i+1}/{inst} ({'echo' if echo else 'forward'}, g={g_value:.3f})", end="\r")
        results = get_single_out(initial_state, i, echo, fixed_g=fixed_g)
        all_results.append(results)
    elapsed = time.time() - start_time
    print(f"\nCompleted {'echo' if echo else 'forward'} simulation with g={g_value:.3f} in {elapsed:.2f}s")
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

# Run real-time adaptive simulation with feedback control
autocorr_adaptive, autocorr_echo_adaptive, g_values_adaptive = get_instances_adaptive_realtime(state)
av_autocorr_adaptive = np.mean(autocorr_adaptive, axis=0)
av_autocorr_echo_adaptive = np.mean(autocorr_echo_adaptive, axis=0)
av_g_values = np.mean(g_values_adaptive, axis=0)

# Also run standard simulations for comparison with different fixed g values
print("\n" + "="*60)
print("Running standard simulations for comparison...")

# Test with g = 0.84 (initial value)
print(f"\n1. Fixed g = {g_initial}")
autocorr_standard_84 = get_instances(state, echo=False, fixed_g=g_initial)
av_autocorr_standard_84 = np.mean(autocorr_standard_84, axis=0)
autocorr_echo_standard_84 = get_instances(state, echo=True, fixed_g=g_initial)
av_autocorr_echo_standard_84 = np.mean(autocorr_echo_standard_84, axis=0)

# Test with g = 0.97 (higher value)
g_high = 0.97
print(f"\n2. Fixed g = {g_high}")
autocorr_standard_97 = get_instances(state, echo=False, fixed_g=g_high)
av_autocorr_standard_97 = np.mean(autocorr_standard_97, axis=0)
autocorr_echo_standard_97 = get_instances(state, echo=True, fixed_g=g_high)
av_autocorr_echo_standard_97 = np.mean(autocorr_echo_standard_97, axis=0)

# For backward compatibility, keep the original variables pointing to g_initial results
autocorr_standard = autocorr_standard_84
av_autocorr_standard = av_autocorr_standard_84
autocorr_echo_standard = autocorr_echo_standard_84
av_autocorr_echo_standard = av_autocorr_echo_standard_84

# Save results with envelope data
try:
    # Calculate envelopes for all data
    upper_env_adaptive_f, lower_env_adaptive_f = find_envelope(av_autocorr_adaptive, window_size=3)
    upper_env_84_f, lower_env_84_f = find_envelope(av_autocorr_standard_84, window_size=3)
    upper_env_97_f, lower_env_97_f = find_envelope(av_autocorr_standard_97, window_size=3)
    
    upper_env_adaptive_e, lower_env_adaptive_e = find_envelope(av_autocorr_echo_adaptive, window_size=3)
    upper_env_84_e, lower_env_84_e = find_envelope(av_autocorr_echo_standard_84, window_size=3)
    upper_env_97_e, lower_env_97_e = find_envelope(av_autocorr_echo_standard_97, window_size=3)
    
    print("Calculated envelopes for data saving")
except Exception as e:
    print(f"Could not calculate envelopes for saving: {e}")
    # Set to None if envelope calculation fails
    upper_env_adaptive_f = lower_env_adaptive_f = None
    upper_env_84_f = lower_env_84_f = None
    upper_env_97_f = lower_env_97_f = None
    upper_env_adaptive_e = lower_env_adaptive_e = None
    upper_env_84_e = lower_env_84_e = None
    upper_env_97_e = lower_env_97_e = None

data = {
    'time': ts,
    'av_autocorr_adaptive': av_autocorr_adaptive,
    'av_autocorr_echo_adaptive': av_autocorr_echo_adaptive,
    'av_g_values': av_g_values,
    'av_autocorr_standard_g84': av_autocorr_standard_84,
    'av_autocorr_echo_standard_g84': av_autocorr_echo_standard_84,
    'av_autocorr_standard_g97': av_autocorr_standard_97,
    'av_autocorr_echo_standard_g97': av_autocorr_echo_standard_97,
    'sqrt_av_autocorr_echo_adaptive': np.sqrt(np.abs(av_autocorr_echo_adaptive)),
    'sqrt_av_autocorr_echo_standard_g84': np.sqrt(np.abs(av_autocorr_echo_standard_84)),
    'sqrt_av_autocorr_echo_standard_g97': np.sqrt(np.abs(av_autocorr_echo_standard_97))
}

# Add envelope data if available
if upper_env_adaptive_f is not None:
    data.update({
        'upper_env_adaptive_forward': upper_env_adaptive_f,
        'lower_env_adaptive_forward': lower_env_adaptive_f,
        'upper_env_g84_forward': upper_env_84_f,
        'lower_env_g84_forward': lower_env_84_f,
        'upper_env_g97_forward': upper_env_97_f,
        'lower_env_g97_forward': lower_env_97_f,
        'upper_env_adaptive_echo': upper_env_adaptive_e,
        'lower_env_adaptive_echo': lower_env_adaptive_e,
        'upper_env_g84_echo': upper_env_84_e,
        'lower_env_g84_echo': lower_env_84_e,
        'upper_env_g97_echo': upper_env_97_e,
        'lower_env_g97_echo': lower_env_97_e
    })
    print("Added envelope data to save dictionary")

# Add individual g histories and comparison data for each instance
for i in range(inst):
    data[f'g_history_inst{i+1}'] = g_values_adaptive[i]
    data[f'echo_adaptive_inst{i+1}'] = autocorr_echo_adaptive[i]
    data[f'forward_adaptive_inst{i+1}'] = autocorr_adaptive[i]
    data[f'echo_standard_g84_inst{i+1}'] = autocorr_echo_standard_84[i]
    data[f'forward_standard_g84_inst{i+1}'] = autocorr_standard_84[i]
    data[f'echo_standard_g97_inst{i+1}'] = autocorr_echo_standard_97[i]
    data[f'forward_standard_g97_inst{i+1}'] = autocorr_standard_97[i]

df = pd.DataFrame(data)
if use_optimization:
    method_suffix = f"_optimization_iter{optimization_iterations}"
else:
    method_suffix = f"_exp{decay_compensation}" if exponential_feedback else "_linear"
csv_filename = f"autocorr_data_{state}_realtime_adaptive{method_suffix}_g{g_initial}_L{L}_inst{inst}_randomphi{args.randomphi}_delta{phi_delta}_amplitude{phi_amplitude}_noise{noise_prob}_usenoise{use_noise}_target{target_echo}_gain{feedback_gain}.csv"

csv_path = f"{folder_name}/{csv_filename}"
df.to_csv(csv_path, index=False)
print(f"Autocorrelation data saved to {csv_path}")

# Save detailed comparison data to separate file
if use_optimization:
    method_short = "optimization"
else:
    method_short = "exponential" if exponential_feedback else "linear"
    
comparison_data = {
    'time': ts,
    'av_g_values': av_g_values,
    'av_echo_adaptive': av_autocorr_echo_adaptive,
    'av_echo_g84': av_autocorr_echo_standard_84,
    'av_echo_g97': av_autocorr_echo_standard_97,
    'av_forward_adaptive': av_autocorr_adaptive,
    'av_forward_g84': av_autocorr_standard_84,
    'av_forward_g97': av_autocorr_standard_97
}

for i in range(inst):
    comparison_data[f'inst{i+1}_g_values'] = g_values_adaptive[i]
    comparison_data[f'inst{i+1}_echo_adaptive'] = autocorr_echo_adaptive[i]
    comparison_data[f'inst{i+1}_echo_g84'] = autocorr_echo_standard_84[i]
    comparison_data[f'inst{i+1}_echo_g97'] = autocorr_echo_standard_97[i]

comp_df = pd.DataFrame(comparison_data)
comp_csv_filename = f"comparison_{state}_adaptive_{method_short}_vs_fixed_g{g_initial}_L{L}_inst{inst}_target{target_echo}_gain{feedback_gain}.csv"
comp_csv_path = f"{folder_name}/{comp_csv_filename}"
comp_df.to_csv(comp_csv_path, index=False)
print(f"Comparison data saved to {comp_csv_path}")

# Create comprehensive comparison plots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(5.7*3, 4.3*2))

# Define consistent colors for each method
# color_adaptive = '#d62728'      # Blue for adaptive
# color_g84 = '#1f77b4'           # Orange for g=0.84
# color_g97 = '#000000'           # Green for g=0.97
# color_target = "#400568"        # Red for target line
# color_bounds = '#8c564b'        # Brown for g bounds
color_adaptive = color_palette[0]      # Blue for adaptive
color_g84 = color_palette[1]           # Orange for g=0.84
color_g97 = color_palette[2]           # Green for g=0.97
color_target = color_palette[3]        # Red for target line
color_bounds = color_palette[4]        # Brown for g bounds

# Plot 1: Forward evolution comparison - all three methods with envelopes
ax1.plot(ts, av_autocorr_adaptive, color=color_adaptive, linestyle='-', label=rf"$A$ (Adaptive)", linewidth=2.5, alpha=0.9)
ax1.plot(ts, av_autocorr_standard_84, color=color_g84, linestyle='--', label=rf"$A$ (Fixed g=0.84)", linewidth=2, alpha=0.8)
ax1.plot(ts, av_autocorr_standard_97, color=color_g97, linestyle='-.', label=rf"$A$ (Fixed g=0.97)", linewidth=2, alpha=0.8)

# Add envelope fits for forward evolution
try:
    # Adaptive envelope
    upper_env_adaptive, lower_env_adaptive = find_envelope(av_autocorr_adaptive, window_size=3)
    ax1.fill_between(ts, lower_env_adaptive, upper_env_adaptive, alpha=0.2, color=color_adaptive)
    
    # Fixed g=0.84 envelope  
    upper_env_84, lower_env_84 = find_envelope(av_autocorr_standard_84, window_size=3)
    ax1.fill_between(ts, lower_env_84, upper_env_84, alpha=0.15, color=color_g84)
    
    # Fixed g=0.97 envelope
    upper_env_97, lower_env_97 = find_envelope(av_autocorr_standard_97, window_size=3)
    ax1.fill_between(ts, lower_env_97, upper_env_97, alpha=0.15, color=color_g97)
    
    print("Added envelope fits to forward evolution plot")
except Exception as e:
    print(f"Could not add envelope fits to forward evolution: {e}")

# Plot only first instance as thin line for adaptive to reduce clutter
# if inst > 0:
#     ax1.plot(ts, autocorr_adaptive[0], color=color_adaptive, linestyle=':', alpha=0.4, linewidth=1, label="Instance 1")
ax1.set_xlabel("t (FT)")
ax1.set_ylabel(r"$\langle Z(0) Z(t) \rangle$")
ax1.set_title("Autocorrelation Evolution")
ax1.legend(loc='best', framealpha=0.9)
ax1.grid(True, alpha=0.3)

# Plot 2: Echo evolution comparison - all three methods with envelopes
ax2.plot(ts, av_autocorr_echo_adaptive, color=color_adaptive, linestyle='-', label=rf"$A_0$", linewidth=2.5, alpha=0.9)
ax2.plot(ts, av_autocorr_echo_standard_84, color=color_g84, linestyle='-', label=rf"$A_0$ (Fixed g=0.84)", linewidth=2, alpha=0.8)
ax2.plot(ts, av_autocorr_echo_standard_97, color=color_g97, linestyle='-.', label=rf"$A_0$ (Fixed g=0.97)", linewidth=2, alpha=0.8)

# Add envelope fits for echo evolution
try:
    # Adaptive echo envelope
    upper_env_echo_adaptive, lower_env_echo_adaptive = find_envelope(av_autocorr_echo_adaptive, window_size=3)
    ax2.fill_between(ts, lower_env_echo_adaptive, upper_env_echo_adaptive, alpha=0.2, color=color_adaptive)
    
    # Fixed g=0.84 echo envelope
    upper_env_echo_84, lower_env_echo_84 = find_envelope(av_autocorr_echo_standard_84, window_size=3)
    ax2.fill_between(ts, lower_env_echo_84, upper_env_echo_84, alpha=0.15, color=color_g84)
    
    # Fixed g=0.97 echo envelope
    upper_env_echo_97, lower_env_echo_97 = find_envelope(av_autocorr_echo_standard_97, window_size=3)
    ax2.fill_between(ts, lower_env_echo_97, upper_env_echo_97, alpha=0.15, color=color_g97)
    
    print("Added envelope fits to echo evolution plot")
except Exception as e:
    print(f"Could not add envelope fits to echo evolution: {e}")

# Plot only first instance as thin line for adaptive to reduce clutter
# if inst > 0:
    # ax2.plot(ts, autocorr_echo_adaptive[0], color=color_adaptive, linestyle=':', alpha=0.4, linewidth=1, label="Instance 1")
# ax2.axhline(y=target_echo, color=color_target, linestyle=':', alpha=0.7, linewidth=1.5,
#             #  label=f'Target ({target_echo})'
#                 )
ax2.set_xlabel("t (FT)")
ax2.set_ylabel(r"$\langle Z(0) Z(t) \rangle$")
ax2.set_title("Echo Evolution")
ax2.legend(loc='best', framealpha=0.9)
ax2.grid(True, alpha=0.3)

# Plot 3: Adaptive g values over time with individual instances
ax3.plot(ts, av_g_values, color=color_adaptive, linestyle='-', label=r"$\bar{g(t)}$", linewidth=3)

# Fill area under the curve where g < 0.97
g_97_value = 0.97
# Create a mask for where g values are less than 0.97
mask_below_97 = av_g_values < g_97_value
# Fill the area under the curve where g < 0.97, above the baseline g_initial (0.84)
ax3.fill_between(ts, g_initial, av_g_values, where=mask_below_97, 
                 color=color_adaptive, alpha=0.3, interpolate=True, 
                #  label='Area where g < 0.97'
                 )

# Plot individual g trajectories
# for i in range(min(inst, 5)):  # Plot max 5 instances
#     ax3.plot(ts, g_values_adaptive[i], color=color_adaptive, linestyle=':', alpha=0.4, linewidth=1, label=f"Instance {i+1}" if i < 3 else None)

# Add horizontal reference lines
ax3.axhline(y=g_initial, color='black', linestyle='--', alpha=0.6, 
            # label=f'Initial g ({g_initial})'
            )
ax3.axhline(y=0.97, color=color_g97, linestyle='-', alpha=0.8, linewidth=2,
#  label=f'g=0.97'
)
ax3.axhline(y=g_min, color=color_bounds, linestyle=':', alpha=0.6,
#  label=f'g_min ({g_min})'
)
ax3.axhline(y=g_max, color=color_bounds, linestyle=':', alpha=0.6,
#  label=f'g_max ({g_max})'
)

# Add shading around g=0.97
y_min, y_max = ax3.get_ylim()
# Shade below g=0.97
ax3.fill_between(ts, y_min, g_97_value, alpha=0.1, color=color_g97, 
                #  label='Below g=0.97'
                 )
# Shade above g=0.97  
ax3.fill_between(ts, g_97_value, y_max, alpha=0.1, color=color_adaptive,
                # label='Above g=0.97'
                )

ax3.set_xlabel("t (FT)")
ax3.set_ylabel("g value")
ax3.set_title("g Parameter Evolution")
ax3.set_yticks(np.arange(0.84, 1.0, 0.02))  # Show every 5th time step for clarity
ax3.legend()
ax3.grid(True, alpha=0.3)

# plt.tight_layout()
if use_optimization:
    method_name = f"Optimization (iter={optimization_iterations})"
    method_short = "optimization"
else:
    method_name = "Exponential" if exponential_feedback else "Linear"
    method_short = method_name.lower()
plt.suptitle(f"Adaptive vs Fixed g Comparison ({method_name}) L={L}, g∈[{g_min}, {g_max}]", y=0.98)

# Save the plot
plot_filename = f"adaptive_vs_fixed_g_comparison_{method_short}_{state}_L{L}_inst{inst}_target{target_echo}_gain{feedback_gain}.png"
plot_path = f"{folder_name}/{plot_filename}"
# plt.savefig(plot_path, dpi=600, bbox_inches='tight')
plt.savefig(plot_path, dpi=600)
print(f"Comparison plot saved to {plot_path}")

plt.show()

# Print comprehensive comparison statistics
print("\n" + "="*70)
print("COMPREHENSIVE COMPARISON STATISTICS")
print("="*70)

methods = ["Adaptive", "g=0.84", "g=0.97"]
forward_finals = [av_autocorr_adaptive[-1], av_autocorr_standard_84[-1], av_autocorr_standard_97[-1]]
echo_finals = [av_autocorr_echo_adaptive[-1], av_autocorr_echo_standard_84[-1], av_autocorr_echo_standard_97[-1]]
echo_errors = [
    np.mean(np.abs(av_autocorr_echo_adaptive - target_echo)),
    np.mean(np.abs(av_autocorr_echo_standard_84 - target_echo)),
    np.mean(np.abs(av_autocorr_echo_standard_97 - target_echo))
]

print(f"{'Method':<15} {'Final Forward':<15} {'Final Echo':<15} {'Avg Echo Error':<15} {'Echo @ t=0':<15}")
print("-" * 75)
initial_echoes = [av_autocorr_echo_adaptive[0], av_autocorr_echo_standard_84[0], av_autocorr_echo_standard_97[0]]
for i, method in enumerate(methods):
    print(f"{method:<15} {forward_finals[i]:<15.4f} {echo_finals[i]:<15.4f} {echo_errors[i]:<15.4f} {initial_echoes[i]:<15.4f}")

# Add envelope analysis
print(f"\nENVELOPE ANALYSIS:")
print("-" * 40)
try:
    # Calculate envelope widths (upper - lower)
    forward_signals = [av_autocorr_adaptive, av_autocorr_standard_84, av_autocorr_standard_97]
    echo_signals = [av_autocorr_echo_adaptive, av_autocorr_echo_standard_84, av_autocorr_echo_standard_97]
    
    for i, method in enumerate(methods):
        # Forward envelope analysis
        upper_env_f, lower_env_f = find_envelope(forward_signals[i], window_size=3)
        forward_env_width = np.mean(upper_env_f - lower_env_f)
        forward_env_ratio = forward_env_width / (np.max(forward_signals[i]) - np.min(forward_signals[i]))
        
        # Echo envelope analysis
        upper_env_e, lower_env_e = find_envelope(echo_signals[i], window_size=3)
        echo_env_width = np.mean(upper_env_e - lower_env_e)
        echo_env_ratio = echo_env_width / (np.max(echo_signals[i]) - np.min(echo_signals[i]))
        
        print(f"{method:<15}: Forward env width = {forward_env_width:.4f} ({forward_env_ratio:.1%}), Echo env width = {echo_env_width:.4f} ({echo_env_ratio:.1%})")
        
except Exception as e:
    print(f"Envelope analysis failed: {e}")

print(f"\nAdaptive g statistics:")
print(f"  Initial g: {g_initial:.4f}")
print(f"  Final g: {av_g_values[-1]:.4f}")
print(f"  g range: [{np.min(av_g_values):.4f}, {np.max(av_g_values):.4f}]")
print(f"  g std dev: {np.std(av_g_values):.4f}")

print(f"\nBest performance (lowest echo error):")
best_idx = np.argmin(echo_errors)
print(f"  Method: {methods[best_idx]}")
print(f"  Average echo error: {echo_errors[best_idx]:.4f}")
print(f"  Improvement over worst: {(max(echo_errors) - echo_errors[best_idx])/max(echo_errors)*100:.1f}%")

# Calculate echo decay rates for comparison
print(f"\nEcho decay analysis:")
for i, method in enumerate(methods):
    if i == 0:
        echo_vals = av_autocorr_echo_adaptive
    elif i == 1:
        echo_vals = av_autocorr_echo_standard_84
    else:
        echo_vals = av_autocorr_echo_standard_97
    
    if len(echo_vals) > 1 and echo_vals[0] > 0:
        # Simple decay rate: (final - initial) / initial / time
        decay_rate = (echo_vals[-1] - echo_vals[0]) / echo_vals[0] / (len(echo_vals) - 1)
        print(f"  {method}: decay rate = {decay_rate:.4f} per time step")

print("\nData files saved:")
print(f"  Main results: {csv_path}")
print(f"  Comparison data: {comp_csv_path}")