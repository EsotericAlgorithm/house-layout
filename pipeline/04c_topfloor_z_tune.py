# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "matplotlib", "trimesh"]
# ///

"""Top floor Z shift picker — renders XZ side view with a range of downward shifts."""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import subprocess
from config import OUT

cache = OUT / "04b_global_transforms.pkl"
with open(cache, "rb") as f:
    data = pickle.load(f)

global_T  = data["global_T"]
photo_pts = data["photo_pts"]

FLOOR_COLORS = {"basement": "#e8b86d", "main_floor": "#7ec8e3",
                "top_floor": "#b5ead7", "attic": "#c9b1ff"}
BG = "#050510"; S = 6

Z_SHIFTS = [0.0, -0.3, -0.6, -0.9, -1.2, -1.5, -1.8, -2.2]

def apply4(pts, T):
    ones = np.ones((len(pts), 1), dtype=np.float64)
    return (T @ np.hstack([pts, ones]).T).T[:, :3]

ncols = len(Z_SHIFTS)
fig, axes = plt.subplots(1, ncols, figsize=(ncols * 4, 6), facecolor="#0d0d1a")
fig.suptitle("Top floor Z shift — pick best alignment with main floor ceiling (XZ view)",
             color="white", fontsize=11, fontweight="bold")

for ax, dz in zip(axes, Z_SHIFTS):
    ax.set_facecolor(BG)
    ax.tick_params(colors="#444466", labelsize=6)
    for sp in ax.spines.values(): sp.set_edgecolor("#222244")

    for floor, T in global_T.items():
        T_mod = T.copy()
        if floor == "top_floor":
            T_mod[2, 3] += dz
        pts = apply4(photo_pts[floor].astype(np.float64), T_mod)
        ax.scatter(pts[::S, 0], pts[::S, 2],
                   c=FLOOR_COLORS[floor], s=0.3, alpha=0.6,
                   linewidths=0, rasterized=True)

    ax.set_aspect("equal")
    ax.set_title(f"{dz:+.1f}m" if dz else "current",
                 color="white", fontsize=9, pad=3)

out_img = OUT / "04c_topfloor_z_tune.png"
plt.tight_layout()
plt.savefig(out_img, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
subprocess.run(["open", str(out_img)])
print(f"Saved → {out_img}")
