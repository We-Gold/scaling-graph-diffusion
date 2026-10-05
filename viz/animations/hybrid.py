import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import argparse
from pathlib import Path

# CLI. --frames is also the number of diffusion steps (one frame per step).
parser = argparse.ArgumentParser(description="Render hybrid.mp4 (toy discrete diffusion animation)")
parser.add_argument("--frames", type=int, default=100, help="number of frames = diffusion steps")
parser.add_argument("--dpi", type=int, default=300)
parser.add_argument("--seed", type=int, default=0, help="seed for the toy data and noise")
parser.add_argument("--out", type=Path, default=Path("outputs/hybrid.mp4"))
parser.add_argument("--show", action="store_true", help="also open a matplotlib window")
args = parser.parse_args()
args.out.parent.mkdir(parents=True, exist_ok=True)
np.random.seed(args.seed)

# --- Parameters ---
num_tokens = 40
timesteps = args.frames
num_classes = 3          # number of categorical values (excluding mask)
mask_token = -1          # mask state
values = np.arange(num_classes)

# --- Noise and remasking schedule ---
def alpha_t(t):
    # Cosine schedule for DiGress-style corruption (decays quickly)
    return np.cos((t / 1.0) * np.pi / 2) ** 2

def unmask_schedule(t):
    # MD4-style unmasking schedule (dominates - increases over time)
    # Higher values mean more likely to unmask
    # Starts low (mostly masked) and increases (mostly unmasked)
    return 1.0 - np.cos((t / 1.0) * np.pi / 2) ** 2

def digress_flip_prob(t):
    # Probability of DiGress-style token flipping for masked tokens
    # Small probability throughout, allows some categorical exploration
    return 0.1 * (1.0 - t)  # Decreases slightly over time

t_values = np.linspace(0, 1, timesteps)
alpha = alpha_t(t_values)
unmask_probs = unmask_schedule(t_values)
digress_probs = digress_flip_prob(t_values)

# --- True discrete values ---
true_values = np.random.choice(values, num_tokens)

# --- Forward hybrid diffusion (denoising/sampling process) ---
# Start with all tokens masked, gradually unmask them
states = np.zeros((timesteps, num_tokens), dtype=int)
states[0] = np.full(num_tokens, mask_token)  # Start fully masked

for i in range(1, timesteps):
    unmask_prob = unmask_probs[i]
    digress_prob = digress_probs[i]
    
    prev_state = states[i - 1]
    new_state = prev_state.copy()
    
    # For each token, decide its fate
    for j in range(num_tokens):
        current_val = prev_state[j]
        
        if current_val != mask_token:
            # Token is unmasked - small chance of DiGress flip
            if np.random.rand() < digress_prob:
                # DiGress: flip to a DIFFERENT categorical value
                available_values = [v for v in values if v != current_val]
                new_state[j] = np.random.choice(available_values)
            else:
                # Otherwise stay unchanged (MD4 behavior)
                new_state[j] = current_val
        else:
            # Token is masked - apply unmasking process
            rand = np.random.rand()
            
            # MD4 unmasking dominates - unmask to a categorical value
            if rand < unmask_prob:
                new_state[j] = np.random.choice(values)
            # Otherwise stay masked
            else:
                new_state[j] = mask_token
    
    states[i] = new_state

# --- Visualization setup ---
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')

ax.set_title("Hybrid DiGress + Masked Diffusion Process")
ax.set_xlabel("Token index")
ax.set_ylabel("Time step")
ax.set_zlabel("Token value")
ax.set_xlim(0, num_tokens)
ax.set_ylim(0, timesteps)
ax.set_zlim(-1.5, num_classes - 0.5)

def update(frame):
    ax.clear()
    ax.set_title(f"Hybrid Diffusion (Step {frame+1}/{timesteps})")
    ax.set_xlabel("Token index")
    ax.set_ylabel("Time step")
    ax.set_zlabel("Token value")
    ax.set_xlim(0, num_tokens)
    ax.set_ylim(0, timesteps)
    ax.set_zlim(-1.5, num_classes - 0.5)

    xs, ys, zs, cs = [], [], [], []
    for t in range(frame):
        for i in range(num_tokens):
            val = states[t, i]
            xs.append(i)
            ys.append(t)
            zs.append(-0.2 if val == mask_token else val)
            cs.append(np.nan if val == mask_token else val)

    # Split masked vs unmasked for color handling
    xs, ys, zs, cs = np.array(xs), np.array(ys), np.array(zs), np.array(cs)

    masked = np.isnan(cs)
    unmasked = ~masked

    # Plot unmasked tokens using colormap
    ax.scatter(xs[unmasked], ys[unmasked], zs[unmasked],
               c=cs[unmasked], cmap='plasma', s=30)
    # Plot masked tokens in gray
    ax.scatter(xs[masked], ys[masked], zs[masked],
               c='gray', s=30)

    return ax

ani = FuncAnimation(fig, update, frames=timesteps, interval=100)

# Save animation to video file
# Requires ffmpeg: brew install ffmpeg (macOS) or apt-get install ffmpeg (Linux)
ani.save(args.out, writer='ffmpeg', fps=10, dpi=args.dpi)
print(f"Animation saved to {args.out}")

if args.show:
    plt.show()
