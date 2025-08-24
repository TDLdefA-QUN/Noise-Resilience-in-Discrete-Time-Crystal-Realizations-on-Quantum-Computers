import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
shots = 1024

# with open('jobs-20250731-151439-UTC.json', 'r') as f:
#     data = json.load(f)
with open('autocorr-iqm-data-merged.json', 'r') as f:
    data = json.load(f)

# Sort data by creation time (from earliest to latest)
data.sort(key=lambda x: datetime.fromisoformat(x['created'].replace('Z', '+00:00')))

with open('autocorr-iqm-echo-data-merged.json', 'r') as f:
    data2 = json.load(f)

# Sort data by creation time (from earliest to latest)
data2.sort(key=lambda x: datetime.fromisoformat(x['created'].replace('Z', '+00:00')))


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

expvals = []
# Process data in groups of 20 (each group = 1 instance = 20 seconds)
for i in range(0, len(data), 20):
    instance_data = data[i:i+20]  # Get 20 items for this instance
    instance_expvals = []
    
    for item in instance_data:
        if item["status"] == "completed":
            meas = item['measurements']
            x = np.array(meas[0]["c_1_0_0"])
            x = x.reshape(1, -1)
            c1 = x[0].sum()
            c0 = shots - c1
            bs = {"0": c0, "1": c1}
            expval = compute_z_expectation(bs, 1)
            instance_expvals.append(expval[0])  # Take first element since it's a single qubit
    print(len(instance_expvals))
    # Compute average expectation value for this instance
    # if instance_expvals:  # Only if there are completed jobs in this instance
    # plt.plot(instance_expvals)
    expvals.append(instance_expvals)

expvals2 = []
# Process data in groups of 20 (each group = 1 instance = 20 seconds)
for i in range(0, len(data2), 20):
    instance_data = data2[i:i+20]  # Get 20 items for this instance
    instance_expvals = []
    
    for item in instance_data:
        if item["status"] == "completed":
            meas = item['measurements']
            x = np.array(meas[0]["c_1_0_0"])
            x = x.reshape(1, -1)
            c1 = x[0].sum()
            c0 = shots - c1
            bs = {"0": c0, "1": c1}
            expval = compute_z_expectation(bs, 1)
            instance_expvals.append(expval[0])  # Take first element since it's a single qubit
    print(len(instance_expvals))
    # Compute average expectation value for this instance
    # if instance_expvals:  # Only if there are completed jobs in this instance
    # plt.plot(instance_expvals)
    expvals2.append(instance_expvals)



avg_expvals = np.array(expvals).mean(axis=0)
avg_expvals2 = np.array(expvals2).mean(axis=0)

# Set figure size to A4 (8.27 x 11.69 inches)
plt.figure(figsize=( 11.69, 8.27))

plt.plot(avg_expvals,label="auto correlation")
plt.plot(avg_expvals2,label="echo")
plt.plot(np.sqrt(avg_expvals2), label="sqrt(echo)")
plt.xlabel('t')
plt.ylabel('Expectation Value')
plt.title("IQM Autocorrelation vs Echo")
# plt.title('Average Expectation Values per Instance')
plt.ylim(-1.05, 1.05)
plt.legend()
plt.savefig('autocorr_iqm_comparison.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'autocorr_iqm_comparison.png'")
plt.show()