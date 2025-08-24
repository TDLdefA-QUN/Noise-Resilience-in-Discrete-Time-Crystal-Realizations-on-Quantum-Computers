from qiskit_ibm_runtime.fake_provider import FakeTorino,FakeBrisbane
from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager, Layout
from qiskit.visualization import plot_circuit_layout, plot_gate_map
import matplotlib.pyplot as plt

qubit_coordinates_map = {}
qubit_coordinates_map[133] = [
        [0, 0],
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [0, 5],
        [0, 6],
        [0, 7],
        [0, 8],
        [0, 9],
        [0, 10],
        [0, 11],
        [0, 12],
        [0, 13],
        [0, 14],

        [1, 0],
        [1, 4],
        [1, 8],
        [1, 12],

        [2, 0],
        [2, 1],
        [2, 2],
        [2, 3],
        [2, 4],
        [2, 5],
        [2, 6],
        [2, 7],
        [2, 8],
        [2, 9],
        [2, 10],
        [2, 11],
        [2, 12],
        [2, 13],
        [2, 14],

        [3, 2],
        [3, 6],
        [3, 10],
        [3, 14],

        [4, 0],
        [4, 1],
        [4, 2],
        [4, 3],
        [4, 4],
        [4, 5],
        [4, 6],
        [4, 7],
        [4, 8],
        [4, 9],
        [4, 10],
        [4, 11],
        [4, 12],
        [4, 13],
        [4, 14],

        [5, 0],
        [5, 4],
        [5, 8],
        [5, 12],

        [6, 0],
        [6, 1],
        [6, 2],
        [6, 3],
        [6, 4],
        [6, 5],
        [6, 6],
        [6, 7],
        [6, 8],
        [6, 9],
        [6, 10],
        [6, 11],
        [6, 12],
        [6, 13],
        [6, 14],

        [7, 2],
        [7, 6],
        [7, 10],
        [7, 14],

        [8, 0],
        [8, 1],
        [8, 2],
        [8, 3],
        [8, 4],
        [8, 5],
        [8, 6],
        [8, 7],
        [8, 8],
        [8, 9],
        [8, 10],
        [8, 11],
        [8, 12],
        [8, 13],
        [8, 14],

        [9, 0],
        [9, 4],
        [9, 8],
        [9, 12],

        [10, 0],
        [10, 1],
        [10, 2],
        [10, 3],
        [10, 4],
        [10, 5],
        [10, 6],
        [10, 7],
        [10, 8],
        [10, 9],
        [10, 10],
        [10, 11],
        [10, 12],
        [10, 13],
        [10, 14],

        [11, 2],
        [11, 6],
        [11, 10],
        [11, 14],

        [12, 0],
        [12, 1],
        [12, 2],
        [12, 3],
        [12, 4],
        [12, 5],
        [12, 6],
        [12, 7],
        [12, 8],
        [12, 9],
        [12, 10],
        [12, 11],
        [12, 12],
        [12, 13],
        [12, 14],

        [13, 0],
        [13, 4],
        [13, 8],
        [13, 12],
    ]

qubit_coordinates = qubit_coordinates_map[133]
print(len(qubit_coordinates))

qc = QuantumCircuit(133, 1)
for i in range(1,132):
    qc.rzz(0.1, i, i + 1)
qc.cz(0, int(133/2)+1)

qc.measure(0, 0)
backend = FakeTorino()
# backend = FakeBrisbane()
circ_qubits = [qc.qubits[i] for i in range(133)]
snake_layout = [
    74,20,19,15,0,1,2,3,4,16,5,6,7,8,17,9,10,11,12,13,14,
    18,31,32,33,37,52,51,50,56,49,48,47,36,29,30,28,27,26,25,35,24,23,22,21,34,40,41,
    39,38,53,57,58,59,72,60,61,62,54,42,43,44,45,46,55,65,64,66,67,68,69,70,71,75,90,89,
    88,94,87,86,85,84,93,83,82,73,63,81,80,92,79,78,77,76,91,95,96,97,110,98,99,100,101,
    111,102,103,104,105,112,106,107,108,109,113,128,127,126,132,125,124,123,122,131,121,120,119,118,130,117,116,115,114,129
]
for i in range(133):
    if i not in snake_layout:
        print(f"Qubit {i} is not in the snake layout.")

cmap = backend.coupling_map
print(len(snake_layout))

def draw_torino_layout_with_snake_indices():
    """Draw the Torino qubit layout with snake layout indices as labels"""
    import matplotlib.pyplot as plt
    import networkx as nx
    import matplotlib.cm as cm
    import numpy as np
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    fig.patch.set_facecolor('#f4f4f4')
    ax.set_facecolor('#f4f4f4')
    
    # Create a mapping from physical qubit to snake layout index
    physical_to_snake = {}
    for snake_idx, physical_qubit in enumerate(snake_layout):
        physical_to_snake[physical_qubit] = snake_idx
    
    # Convert qubit coordinates to a more convenient format
    # Scale coordinates for better visualization
    scale_x = 50
    scale_y = 50
    pos = {}
    for i, coord in enumerate(qubit_coordinates):
        x, y = coord
        pos[i] = (x * scale_x, -y * scale_y)  # Negative y to flip vertically
    
    # Create NetworkX graph for the coupling map
    G = nx.Graph()
    G.add_nodes_from(range(133))
    
    # Add edges from coupling map
    for edge in cmap.get_edges():
        G.add_edge(edge[0], edge[1])
    
    # Draw edges (connections) with color and width based on snake index proximity
    for edge in G.edges():
        qubit1, qubit2 = edge
        if qubit1 in pos and qubit2 in pos:
            x1, y1 = pos[qubit1]
            x2, y2 = pos[qubit2]
            
            # Get snake layout indices
            snake_idx1 = physical_to_snake.get(qubit1, -1)
            snake_idx2 = physical_to_snake.get(qubit2, -1)
            
            if snake_idx1 >= 0 and snake_idx2 >= 0:
                # Calculate the difference in snake indices
                idx_diff = abs(snake_idx1 - snake_idx2)
                
                # Determine color and width based on proximity
                if idx_diff == 1:
                    # Adjacent in snake layout - thick red line
                    color = 'red'
                    width = 3.5
                    alpha = 0.8
                elif idx_diff == 2:
                    # 2 steps apart - more contrasting orange line
                    color = 'darkgreen'
                    width = 3
                    alpha = 0.7
                elif idx_diff == 3:
                    # 3 steps apart - yellow line
                    color = 'blue'
                    width = 2.5
                    alpha = 0.6
                else:
                    # 4 or more steps apart - thin blue line
                    color = 'black'
                    width = 2
                    alpha = 0.4
                    
                ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, 
                       linewidth=width, zorder=1)
            else:
                # Default for qubits without snake indices
                ax.plot([x1, x2], [y1, y2], 'gray', alpha=0.3, linewidth=1, zorder=1)
    
    # Create colormap for gradient colors
    colormap = cm.get_cmap('viridis')  # You can use 'plasma', 'inferno', 'cool', 'hot', etc.
    
    # Draw qubits as circles with gradient colors
    for physical_qubit in range(133):
        if physical_qubit in pos:
            x, y = pos[physical_qubit]
            
            # Get snake layout index for this physical qubit
            snake_idx = physical_to_snake.get(physical_qubit, -1)
            
            if snake_idx >= 0:
                # Calculate color based on position in snake layout (0 to 1)
                color_value = snake_idx / (len(snake_layout) - 1)
                color = colormap(color_value)
                
                # Draw circle with gradient color
                circle = plt.Circle((x, y), 15, color=color, 
                                  linewidth=2, edgecolor='darkblue', zorder=2)
                ax.add_patch(circle)
                
                # Add snake layout index as label
                # Use white text for darker colors, black for lighter colors
                text_color = 'white' if color_value < 0.6 else 'black'
                ax.text(x, y, str(snake_idx), ha='center', va='center',
                       fontsize=8, fontweight='bold', color=text_color, zorder=3)
            else:
                # This shouldn't happen if snake_layout is complete
                circle = plt.Circle((x, y), 15, color='red', 
                                  linewidth=2, edgecolor='darkred', zorder=2)
                ax.add_patch(circle)
                ax.text(x, y, '?', ha='center', va='center',
                       fontsize=8, fontweight='bold', color='white', zorder=3)
    
    # Draw arrows for snake layout adjacency (non-physically adjacent qubits)
    for i in range(1, len(snake_layout) - 1):
        current_physical = snake_layout[i]
        next_physical = snake_layout[i + 1]
        
        # Check if these qubits are physically connected in the coupling map
        is_physically_connected = False
        for edge in cmap.get_edges():
            if (edge[0] == current_physical and edge[1] == next_physical) or \
               (edge[0] == next_physical and edge[1] == current_physical):
                is_physically_connected = True
                break
        
        # If they are adjacent in snake layout but NOT physically connected, draw an arrow
        if not is_physically_connected and current_physical in pos and next_physical in pos:
            x1, y1 = pos[current_physical]
            x2, y2 = pos[next_physical]
            
            # Draw a curved arrow to show the snake layout connection
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', color='purple', alpha=0.7, 
                                     lw=2, shrinkA=15, shrinkB=15,
                                     connectionstyle="arc3,rad=0.3"), zorder=4)
    
    # Set axis properties
    ax.set_aspect('equal')
    # ax.set_title('IBM Torino QPU Layout\n(Numbers show Snake Layout indices - Gradient shows flow)', 
    ax.set_title('IBM Torino QPU Layout for Autocorrelation Circuit', 
                fontsize=16, fontweight='bold', pad=10)
    
    # Add legend for edge colors
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch
    legend_elements = [
        Line2D([0], [0], color='red', lw=3, label='Adjacent (diff=1)'),
        Line2D([0], [0], color='darkgreen', lw=2.5, label='2 steps apart (diff=2)'),
        Line2D([0], [0], color='blue', lw=2, label='3 steps apart (diff=3)'),
        Line2D([0], [0], color='black', lw=1, label='4+ steps apart (diff≥4)'),
        Line2D([0], [0], color='purple', lw=2, label='Snake adjacency (non-physical)', 
               linestyle='--', marker='>')
    ]
    # ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5))
    # ax.legend()
    
    # Add colorbar to show the snake layout flow
    # sm = cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0, vmax=len(snake_layout)-1))
    # sm.set_array([])
    # cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=30)
    # cbar.set_label('Snake Layout Index (Flow Direction)', rotation=270, labelpad=20, fontsize=12)
    
    # Add some statistics
    stats_text = f"Total Qubits: {len(snake_layout)}\nTotal Connections: {len(cmap.get_edges())}"
    # ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
    #        verticalalignment='top', fontsize=12,
    #        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add grid for reference (optional)
    # ax.grid(True, alpha=0.2)
    
    # plt.tight_layout()
    return fig, ax

# Draw the layout
fig, ax = draw_torino_layout_with_snake_indices()
plt.savefig('torino-echo-layout.png', dpi=300, bbox_inches='tight')
plt.show()


# gm = plot_gate_map(backend,qubit_coordinates=qubit_coordinates)
# plt.show()

# layout_dict = {circ_qubits[i]: snake_layout[i] for i in range(133)}
# initial_layout = Layout(layout_dict)
# passmanager = generate_preset_pass_manager(
#     optimization_level=3,
#     backend=backend,
#     layout_method="sabre",
#     )

# passmanager2 = generate_preset_pass_manager(
#     optimization_level=3,
#     backend=backend,
#     initial_layout= initial_layout
#     # layout_method="sabre",
#     )


# # print(backend.coupling_map)
# # circ = passmanager.run(qc)
# # gates = circ.count_ops()
# # print(gates)
# # print(sum(gates.values()))

# circ2 = passmanager2.run(qc)
# gates2 = circ2.count_ops()
# print(gates2)
# print(sum(gates2.values()))

# # layout = plot_circuit_layout(circ, backend=backend)
# layout2 = plot_circuit_layout(circ2, backend=backend, qubit_coordinates=qubit_coordinates)

# plt.show()