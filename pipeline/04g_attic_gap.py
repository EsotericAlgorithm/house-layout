# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "matplotlib", "trimesh"]
# ///

"""Quick picker for interstitial gap between top floor ceiling and attic floor."""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import subprocess
from config import OUT

cache = OUT / "04f_final_transforms.pkl"
with open(cache, "rb") as f:
    data = pickle.load(f)

global_T  = data["global_T"]
photo_pts = data["photo_pts"]

FLOOR_COLORS = {"basement": "#e8b86d", "main_floor": "#7ec8e3",
                "top_floor": "#b5ead7", "attic": "#c9b1ff"}
BG = "#050510"; S = 6
GAPS = [0.0, 0.15, 0.3, 0.45, 0.6]

def apply4(pts, T):
    ones = np.ones((len(pts), 1), dtype=np.float64)
    return (T @ np.hstack([pts, ones]).T).T[:, :3]

ref = {f: apply4(photo_pts[f].astype(np.float64), global_T[f])
       for f in ["basement", "main_floor", "top_floor"]}

fig, axes = plt.subplots(1, len(GAPS), figsize=(len(GAPS)*4, 6), facecolor="#0d0d1a")
fig.suptitle("Attic interstitial gap (metres above top floor ceiling)",
             color="white", fontsize=11, fontweight="bold")

for ax, gap in zip(axes, GAPS):
    ax.set_facecolor(BG)
    ax.tick_params(colors="#444466", labelsize=6)
    for sp in ax.spines.values(): sp.set_edgecolor("#222244")
    for f, pts in ref.items():
        ax.scatter(pts[::S, 0], pts[::S, 2], c=FLOOR_COLORS[f],
                   s=0.2, alpha=0.5, linewidths=0, rasterized=True)
    T_attic = global_T["attic"].copy()
    T_attic[2, 3] += gap
    attic_pts = apply4(photo_pts["attic"].astype(np.float64), T_attic)
    ax.scatter(attic_pts[::S, 0], attic_pts[::S, 2], c=FLOOR_COLORS["attic"],
               s=0.3, alpha=0.7, linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    ax.set_title(f"+{gap:.2f}m" if gap else "flush (0m)", color="white", fontsize=9)

out_img = OUT / "04g_attic_gap.png"
plt.tight_layout()
plt.savefig(out_img, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
subprocess.run(["open", str(out_img)])
print(f"Saved → {out_img}")
