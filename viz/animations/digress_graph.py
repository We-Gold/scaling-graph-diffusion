import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import argparse
from pathlib import Path

# CLI. --frames is also the number of diffusion steps (one frame per step).
parser = argparse.ArgumentParser(description="Render digress_graph.mp4 (toy discrete diffusion animation)")
parser.add_argument("--frames", type=int, default=100, help="number of frames = diffusion steps")
parser.add_argument("--dpi", type=int, default=300)
parser.add_argument("--seed", type=int, default=0, help="seed for the toy data and noise")
parser.add_argument("--out", type=Path, default=Path("outputs/digress_graph.mp4"))
parser.add_argument("--show", action="store_true", help="also open a matplotlib window")
args = parser.parse_args()
args.out.parent.mkdir(parents=True, exist_ok=True)
np.random.seed(args.seed)

# --- Parameters ---
num_nodes = 10
num_node_types = 3
num_edge_types = 3  # 0 = no edge, 1 = edge type 1, 2 = edge type 2

timesteps = args.frames

# Probability distributions for node and edge types
# Must sum to 1.0 for each
node_type_probs = [0.5, 0.3, 0.2]  # Probabilities for node types 0, 1, 2
edge_type_probs = [0.75, 0.15, 0.1]  # Probabilities for edge types 0 (no edge), 1, 2

# Validate probabilities
assert len(node_type_probs) == num_node_types, "node_type_probs length must match num_node_types"
assert len(edge_type_probs) == num_edge_types, "edge_type_probs length must match num_edge_types"
assert abs(sum(node_type_probs) - 1.0) < 1e-6, "node_type_probs must sum to 1.0"
assert abs(sum(edge_type_probs) - 1.0) < 1e-6, "edge_type_probs must sum to 1.0"

# --- Cosine noise schedule (DiGress-style) ---
def alpha_t(t):
    return np.cos((t / 1.0) * np.pi / 2) ** 2  # cosine schedule

t_values = np.linspace(0, 1, timesteps)
alpha_nodes = alpha_t(t_values)
alpha_edges = alpha_t(t_values) ** 0.8  # Edges may noise slightly differently

# --- Create ground truth graph ---
true_node_values = np.random.choice(num_node_types, size=num_nodes, p=node_type_probs)
# Edge values: 0 = no edge, 1+ = edge types
# Only generate upper triangle (excluding diagonal) for undirected graph
true_edge_values = np.zeros((num_nodes, num_nodes), dtype=int)
upper_triangle_indices = np.triu_indices(num_nodes, k=1)  # k=1 excludes diagonal
num_upper_triangle_edges = len(upper_triangle_indices[0])
true_edge_values[upper_triangle_indices] = np.random.choice(
    num_edge_types, 
    size=num_upper_triangle_edges, 
    p=edge_type_probs
)

# Make adjacency matrix symmetric for undirected graph
true_edge_values = true_edge_values + true_edge_values.T

# --- Flatten edge matrix to tokens ---
edge_tokens = true_edge_values.flatten()
num_edges = len(edge_tokens)

# --- Forward diffusion process (DiGress style) ---
# Start from true values and gradually add noise
states_nodes = np.zeros((timesteps, num_nodes), dtype=int)
states_edges = np.zeros((timesteps, num_edges), dtype=int)

# Initialize with true values
states_nodes[0] = true_node_values
states_edges[0] = edge_tokens

# Apply noise over time with decreasing flip probability
for i in range(1, timesteps):
    # Calculate flip probability that decreases over time
    # Early timesteps: high flip probability, Late timesteps: low flip probability
    time_factor = 1 - (i / timesteps)  # Goes from 1.0 to 0.0
    
    # Nodes: with probability (1 - alpha) * time_factor, flip to random value
    node_base_flip_prob = 1 - alpha_nodes[i]
    node_flip_prob = node_base_flip_prob * time_factor
    node_flip_mask = np.random.rand(num_nodes) < node_flip_prob
    node_noise = np.random.choice(num_node_types, num_nodes, p=node_type_probs)
    states_nodes[i] = np.where(node_flip_mask, node_noise, states_nodes[i-1])
    
    # Edges: with probability (1 - alpha) * time_factor, flip to random value
    edge_base_flip_prob = 1 - alpha_edges[i]
    edge_flip_prob = edge_base_flip_prob * time_factor
    edge_flip_mask = np.random.rand(num_edges) < edge_flip_prob
    edge_noise = np.random.choice(num_edge_types, num_edges, p=edge_type_probs)
    states_edges[i] = np.where(edge_flip_mask, edge_noise, states_edges[i-1])

# --- Reshape edge states back to adjacency matrices for visualization ---
states_adj = states_edges.reshape(timesteps, num_nodes, num_nodes)

# --- Combine nodes and edges into one token sequence ---
combined_states = np.concatenate([states_nodes, states_edges], axis=1)
num_total_tokens = combined_states.shape[1]

# --- Create fixed node positions for graph visualization ---
np.random.seed(42)  # For consistent layout
pos = {}
angle = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
for i in range(num_nodes):
    pos[i] = (np.cos(angle[i]), np.sin(angle[i]))

# --- Visualization setup ---
fig = plt.figure(figsize=(16, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122)

# Color mapping for nodes
node_colors_palette = ['#FF6B6B', '#4ECDC4', "#CFF473"]  # Red, Teal, Mint
edge_colors_palette = ['#A9A9A9', '#FFA07A', '#6A5ACD']  # Gray (no edge), Light Salmon, Slate Blue

# Color mapping: separate palettes for nodes vs edges
def token_color(idx, val):
    if idx < num_nodes:
        # Node colors
        return node_colors_palette[int(val) % len(node_colors_palette)]
    else:
        # Edge colors
        return edge_colors_palette[int(val) % len(edge_colors_palette)]

def update(frame):
    # Clear both subplots
    ax1.clear()
    ax2.clear()
    
    # --- 3D Token Plot ---
    ax1.set_title(f"DiGress-style Graph Diffusion (Step {frame+1}/{timesteps})", fontsize=12)
    ax1.set_xlabel("Token index (nodes → edges)")
    ax1.set_ylabel("Time step")
    ax1.set_zlabel("Token value")
    ax1.set_xlim(0, num_total_tokens)
    ax1.set_ylim(0, timesteps)
    ax1.set_zlim(-0.5, max(num_node_types, num_edge_types) - 0.5)

    xs, ys, zs, cs = [], [], [], []
    for t in range(frame):
        for i in range(num_total_tokens):
            val = combined_states[t, i]
            xs.append(i)
            ys.append(t)
            zs.append(val)
            cs.append(token_color(i, val))

    ax1.scatter(xs, ys, zs, c=cs, s=30)
    
    # --- Graph Visualization ---
    ax2.set_title(f"Graph Structure at Step {frame+1}/{timesteps}", fontsize=12)
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.axis('off')
    
    # Get current state
    current_nodes = states_nodes[frame-1] if frame > 0 else states_nodes[0]
    current_adj = states_adj[frame-1] if frame > 0 else states_adj[0]
    
    # Draw nodes
    for i in range(num_nodes):
        x, y = pos[i]
        node_val = current_nodes[i]
        color = node_colors_palette[int(node_val) % len(node_colors_palette)]
        
        circle = plt.Circle((x, y), 0.15, color=color, alpha=1.0, zorder=2)
        ax2.add_patch(circle)
        ax2.text(x, y, str(i), ha='center', va='center', fontsize=10, 
                fontweight='bold', zorder=3)
    
    # Draw edges
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):  # Only upper triangle (undirected)
            edge_val = current_adj[i, j]
            if edge_val > 0:  # Edge exists (not type 0)
                x1, y1 = pos[i]
                x2, y2 = pos[j]
                edge_color = edge_colors_palette[int(edge_val) % len(edge_colors_palette)]
                ax2.plot([x1, x2], [y1, y2], color=edge_color, linewidth=2, 
                        alpha=0.7, zorder=1)
    
    # Add legend
    legend_elements = []
    for i, color in enumerate(node_colors_palette[:num_node_types]):
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                      markersize=10, label=f'Node Type {i}')
        )
    legend_elements.append(
        plt.Line2D([0], [0], color='none', label='Edge Type 0: None')
    )
    for i in range(1, num_edge_types):
        color = edge_colors_palette[i]
        legend_elements.append(
            plt.Line2D([0], [0], color=color, linewidth=2, label=f'Edge Type {i}')
        )
    
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    return ax1, ax2

ani = FuncAnimation(fig, update, frames=timesteps, interval=100)

# Save animation to video file
# Requires ffmpeg: brew install ffmpeg (macOS) or apt-get install ffmpeg (Linux)
ani.save(args.out, writer='ffmpeg', fps=10, dpi=args.dpi)
print(f"Animation saved to {args.out}")

if args.show:
    plt.show()
