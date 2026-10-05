import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# --- Parameters ---
num_nodes = 40           # number of tokens/nodes
timesteps = 100          # number of steps
num_classes = 3          # number of possible discrete categories
values = np.arange(num_classes)

# --- Cosine noise schedule (DiGress-style) ---
def alpha_t(t):
    return np.cos((t / 1.0) * np.pi / 2) ** 2  # cosine schedule
t_values = np.linspace(0, 1, timesteps)
alpha = alpha_t(t_values)

# --- True original node labels ---
true_values = np.random.choice(values, num_nodes)

# --- Forward process (DiGress style) ---
# Start from the true labels
states = np.zeros((timesteps, num_nodes), dtype=int)
states[0] = true_values

for i in range(1, timesteps):
    keep_prob = alpha[i]
    random_mask = np.random.rand(num_nodes) > keep_prob
    random_noise = np.random.choice(values, num_nodes)
    states[i] = np.where(random_mask, random_noise, states[i-1])

# --- Visualization setup ---
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')

ax.set_title("DiGress-style Discrete Diffusion (Forward Process)")
ax.set_xlabel("Node index")
ax.set_ylabel("Time step")
ax.set_zlabel("Node value")
ax.set_xlim(0, num_nodes)
ax.set_ylim(0, timesteps)
ax.set_zlim(-0.5, num_classes - 0.5)

def update(frame):
    ax.clear()
    ax.set_title(f"DiGress Forward Diffusion (Step {frame+1}/{timesteps})")
    ax.set_xlabel("Node index")
    ax.set_ylabel("Time step")
    ax.set_zlabel("Node value")
    ax.set_xlim(0, num_nodes)
    ax.set_ylim(0, timesteps)
    ax.set_zlim(-0.5, num_classes - 0.5)

    xs, ys, zs, cs = [], [], [], []
    for t in range(frame):
        xs.extend(np.arange(num_nodes))
        ys.extend([t] * num_nodes)
        zs.extend(states[t])
        cs.extend(states[t])  # color by class index

    ax.scatter(xs, ys, zs, c=cs, cmap='plasma', s=30)
    return ax

ani = FuncAnimation(fig, update, frames=timesteps, interval=100)

# Save animation to video file
# Requires ffmpeg: brew install ffmpeg (macOS) or apt-get install ffmpeg (Linux)
ani.save('digress_animation.mp4', writer='ffmpeg', fps=10, dpi=300)
print("Animation saved to digress_animation.mp4")

plt.show()
