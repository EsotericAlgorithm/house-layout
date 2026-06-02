# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "matplotlib", "trimesh"]
# ///

"""Locks in +0.3m attic gap and saves the definitive global transforms."""

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

global_T      = data["global_T"]
photo_pts     = data["photo_pts"]
lidar_clipped = data["lidar_clipped"]
per_floor_T   = data["per_floor_T"]

global_T["attic"][2, 3] += 0.3

print("Final global transforms:")
for floor, T in global_T.items():
    print(f"  {floor}: X={T[0,3]:.3f}m  Z={T[2,3]:.3f}m")

def apply4(pts, T):
    ones = np.ones((len(pts), 1), dtype=np.float64)
    return (T @ np.hstack([pts, ones]).T).T[:, :3]

FLOOR_COLORS = {"basement": "#e8b86d", "main_floor": "#7ec8e3",
                "top_floor": "#b5ead7", "attic": "#c9b1ff"}
BG = "#050510"; S = 6

fig, axes = plt.subplots(1, 2, figsize=(20, 9), facecolor="#0d0d1a")
fig.suptitle("Final Global Stack — all floors registered",
             color="white", fontsize=13, fontweight="bold")

for ax, (xi, yi, label) in zip(axes, [(0,1,"Top-down (XY)"), (0,2,"Side (XZ)")]):
    ax.set_facecolor(BG)
    ax.set_title(label, color="white", fontsize=10)
    ax.tick_params(colors="#444466", labelsize=6)
    for sp in ax.spines.values(): sp.set_edgecolor("#222244")
    for floor, T in global_T.items():
        pts = apply4(photo_pts[floor].astype(np.float64), T)
        ax.scatter(pts[::S, xi], pts[::S, yi], c=FLOOR_COLORS[floor],
                   s=0.3, alpha=0.6, linewidths=0, rasterized=True, label=floor)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, facecolor="#111122", labelcolor="white",
              edgecolor="#333355", markerscale=8)

out_img = OUT / "04h_final_stack.png"
plt.tight_layout()
plt.savefig(out_img, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
subprocess.run(["open", str(out_img)])
print(f"Saved → {out_img}")

cache_out = OUT / "FINAL_transforms.pkl"
with open(cache_out, "wb") as f:
    pickle.dump({"global_T": global_T, "photo_pts": photo_pts,
                 "lidar_clipped": lidar_clipped, "per_floor_T": per_floor_T}, f)
print(f"Definitive transforms → {cache_out}")
