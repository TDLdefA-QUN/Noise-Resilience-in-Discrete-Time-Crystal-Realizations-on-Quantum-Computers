# This script generates all parameter combinations and writes them to params.csv
from itertools import product

deltas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.5, 2.0]
amps = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.5, 2.0]
gs = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

with open('params.csv', 'w') as f:
    for g, amp, delta in product(gs, amps, deltas):
        f.write(f"{g},{amp},{delta}\n")
