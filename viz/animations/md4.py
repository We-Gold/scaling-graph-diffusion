import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# --- Parameters ---
num_tokens = 40
timesteps = 100
values = np.array([0, 1])
mask_token = -1

# --- Cosine unmasking schedule (MD4 style) ---
def alpha_t(t):
    return 1 - np.cos(np.pi / 2 * (1 - t))

t_values = np.linspace(0, 1, timesteps)
alpha = alpha_t(t_values)
unmask_prob = 1 - alpha  # cumulative fraction expected to be unmasked

# --- True discrete values ---
true_values = np.random.choice(values, num_tokens)

# --- Sample "unmasking times" for each token ---
# Each token gets a random unmask time proportional to the cosine schedule
unmask_time = np.random.rand(num_tokens)

# --- Simulate monotonic unmasking ---
states = np.full((timesteps, num_tokens), mask_token)
for i, t in enumerate(t_values):
    threshold = unmask_prob[i]
    # Tokens whose sampled unmask time < current threshold become visible
    unmasked = unmask_time < threshold
    states[i] = np.where(unmasked, true_values, mask_token)

# --- Calculate empirical unmasking schedule ---
empirical_unmask_fraction = np.zeros(timesteps)
for i in range(timesteps):
    empirical_unmask_fraction[i] = np.mean(states[i] != mask_token)

# --- Visualization setup ---
fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, projection='3d')

# Plot unmasking schedule comparison
ax1.plot(t_values, unmask_prob, 'b-', linewidth=2, label='Theoretical (Cosine)')
ax1.plot(t_values, empirical_unmask_fraction, 'r--', linewidth=2, label='Empirical')
ax1.set_xlabel('Normalized Time (t)', fontsize=11)
ax1.set_ylabel('Unmasked Fraction', fontsize=11)
ax1.set_title('Unmasking Schedule', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)

ax2.set_title("Reverse Masked Diffusion (Monotonic Unmasking, Cosine Schedule)")
ax2.set_xlabel("Token index")
ax2.set_ylabel("Time step")
ax2.set_zlabel("Token value")
ax2.set_xlim(0, num_tokens)
ax2.set_ylim(0, timesteps)
ax2.set_zlim(-0.5, 1.5)

def update(frame):
    ax2.clear()
    ax2.set_title(f"Reverse Masked Diffusion (Step {frame+1}/{timesteps})")
    ax2.set_xlabel("Token index")
    ax2.set_ylabel("Time step")
    ax2.set_zlabel("Token value")
    ax2.set_xlim(0, num_tokens)
    ax2.set_ylim(0, timesteps)
    ax2.set_zlim(-0.5, 1.5)

    xs, ys, zs, cs = [], [], [], []
    for t in range(frame):
        for i in range(num_tokens):
            val = states[t, i]
            xs.append(i)
            ys.append(t)
            if val == mask_token:
                zs.append(-0.2)
                cs.append("gray")  # masked
            else:
                zs.append(val)
                cs.append("red" if val == 1 else "blue")  # unmasked

    ax2.scatter(xs, ys, zs, c=cs, s=30)
    return ax2

ani = FuncAnimation(fig, update, frames=timesteps, interval=100)

# Save animation to video file
# Requires ffmpeg: brew install ffmpeg (macOS) or apt-get install ffmpeg (Linux)
ani.save('md4_animation_2.mp4', writer='ffmpeg', fps=10, dpi=300)
print("Animation saved to md4_animation.mp4")

plt.show()
