import numpy as np

def generate_disorder(L, inst, phi_amplitude=1.0, phi_delta=0.0, randomphi=1):
    """
    Generate random hs and phis arrays for given number of qubits and instances.
    Args:
        L (int): Number of qubits
        inst (int): Number of instances
        phi_amplitude (float): Amplitude parameter for phis
        phi_delta (float): Delta parameter for phis
        randomphi (int): If 1, generate random phis; else use fixed value
    Returns:
        hs (np.ndarray): shape (inst, L)
        phis (np.ndarray): shape (inst, L-1)
    """
    hs = np.random.random((inst, L)) * 2 * np.pi - np.pi  # [-pi, pi]
    if randomphi == 1:
        phis = np.random.random((inst, L-1)) * phi_amplitude * np.pi - 1.5 * np.pi + phi_delta * np.pi  # [-1.5pi, -0.5pi]
    else:
        phis = np.full((inst, L-1), -0.4)
    return hs, phis


def save_disorder_to_csv(L, inst, phi_amplitude=1.0, phi_delta=0.0, randomphi=1, folder="."):
    """
    Generate and save hs and phis arrays to CSV files.
    Args:
        L (int): Number of qubits
        inst (int): Number of instances
        phi_amplitude (float): Amplitude parameter for phis
        phi_delta (float): Delta parameter for phis
        randomphi (int): If 1, generate random phis; else use fixed value
        folder (str): Folder to save CSV files
    """
    import pandas as pd
    hs, phis = generate_disorder(L, inst, phi_amplitude, phi_delta, randomphi)
    hs_filename = f"{folder}/hs_L{L}_inst{inst}_ampl{phi_amplitude}_delta{phi_delta}_randomphi{randomphi}.csv"
    phis_filename = f"{folder}/phis_L{L}_inst{inst}_ampl{phi_amplitude}_delta{phi_delta}_randomphi{randomphi}.csv"
    hs_df = pd.DataFrame(hs)
    phis_df = pd.DataFrame(phis)
    hs_df.to_csv(hs_filename, index=False, header=[f"h_{i}" for i in range(hs.shape[1])])
    phis_df.to_csv(phis_filename, index=False, header=[f"phi_{i}" for i in range(phis.shape[1])])
    print(f"Saved hs to {hs_filename}")
    print(f"Saved phis to {phis_filename}")


# Generate and save disorder for L=4 to L=130
if __name__ == "__main__":
    inst = 3  # You can change this as needed
    phi_amplitude = 1.0
    phi_delta = 0.0
    randomphi = 1
    folder = "disorder_data"
    for L in range(4, 131):
        save_disorder_to_csv(L, inst, phi_amplitude, phi_delta, randomphi, folder)
