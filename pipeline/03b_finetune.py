# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "trimesh", "matplotlib", "scipy"]
# ///

"""
Fine-tunes the main floor registration with a small manual X offset.
Renders a picker row so you can choose the best shift amount.
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import subprocess
from config import OUT

cache = OUT / "03_transforms.pkl"
with open(cache, "rb") as f:
    data = pickle.load(f)

photo_pts     = data["photo"]
lidar_clipped = data["lidar_clipped"]
transforms    = data["transforms"]

# X offsets to try (metres, positive = rightward)
X_SHIFTS = [0.0, 0.15, 0.3, 0.5, 0.75, 1.0]

BG = "#050510"
S  = 5
floor = "main_floor"
photo = photo_pts[floor]
lidar = lidar_clipped[floor]
T     = transforms[floor]

def apply(pts, T):
    ones = np.ones((len(pts), 1))
    return (T @ np.hstack([pts, ones]).T).T[:, :3]

ncols = len(X_SHIFTS)
fig, axes = plt.subplots(1, ncols, figsize=(ncols * 5, 5), facecolor="#0d0d1a")
fig.suptitle("Main floor — pick X shift (rightward, metres)",
             color="white", fontsize=12, fontweight="bold")

for ax, dx in zip(axes, X_SHIFTS):
    T_shifted = T.copy()
    T_shifted[0, 3] += dx
    lidar_reg = apply(lidar.astype(np.float64), T_shifted)

    ax.scatter(photo[::S, 0], photo[::S, 1],
               c="#ffffff", s=0.2, alpha=0.35, linewidths=0, rasterized=True)
    ax.scatter(lidar_reg[::S, 0], lidar_reg[::S, 1],
               c="#7ec8e3", s=0.3, alpha=0.55, linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    ax.set_facecolor(BG)
    ax.set_title(f"+{dx:.2f}m" if dx else "current (0m)",
                 color="white", fontsize=10, pad=4)
    ax.tick_params(colors="#444466", labelsize=6)
    for sp in ax.spines.values(): sp.set_edgecolor("#222244")

out_img = OUT / "03b_finetune_main_floor.png"
plt.tight_layout()
plt.savefig(out_img, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {out_img}")
subprocess.run(["open", str(out_img)])
