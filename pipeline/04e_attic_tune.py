# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "matplotlib", "trimesh"]
# ///

"""
Attic alignment tuner — shows the attic against the full stack in XZ
with different candidate orientations and X positions.
Row 1: orientation candidates (same set as 02b, shown in XZ context)
Row 2: X shift candidates (once orientation is known)
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

cache = OUT / "04d_final_transforms.pkl"
with open(cache, "rb") as f:
    data = pickle.load(f)

global_T  = data["global_T"]
photo_pts = data["photo_pts"]

FLOOR_COLORS = {"basement": "#e8b86d", "main_floor": "#7ec8e3",
                "top_floor": "#b5ead7", "attic": "#c9b1ff"}
BG = "#050510"; S = 6

def apply4(pts, T):
    ones = np.ones((len(pts), 1), dtype=np.float64)
    return (T @ np.hstack([pts, ones]).T).T[:, :3]

# Other floors in global space (fixed reference)
ref_floors = {f: apply4(photo_pts[f].astype(np.float64), global_T[f])
              for f in ["basement", "main_floor", "top_floor"]}

attic_raw = photo_pts["attic"].astype(np.float64)
T_attic   = global_T["attic"].copy()

# ── Row 1: orientation candidates ────────────────────────────────────────────
ORIENT = {
    "current":        np.eye(3),
    "flip X+Y":       np.diag([-1.,-1.,1.]),
    "flip X":         np.diag([-1., 1.,1.]),
    "flip Y":         np.diag([ 1.,-1.,1.]),
    "swap X↔Y":       np.array([[0,1,0],[1,0,0],[0,0,1]], float),
    "swap+flip X+Y":  np.array([[0,-1,0],[-1,0,0],[0,0,1]], float),
}

# ── Row 2: X shifts (applied on top of current orientation) ──────────────────
X_SHIFTS = [0, 1, 2, 3, 4, 5, 6, 7]

ncols = max(len(ORIENT), len(X_SHIFTS))
fig, axes = plt.subplots(2, ncols, figsize=(ncols * 3.5, 10), facecolor="#0d0d1a")
fig.suptitle("Attic alignment — Row 1: orientation  |  Row 2: X shift (metres)",
             color="white", fontsize=11, fontweight="bold")

def draw_stack(ax, attic_pts, title):
    for f, pts in ref_floors.items():
        ax.scatter(pts[::S, 0], pts[::S, 2], c=FLOOR_COLORS[f],
                   s=0.2, alpha=0.5, linewidths=0, rasterized=True)
    ax.scatter(attic_pts[::S, 0], attic_pts[::S, 2], c=FLOOR_COLORS["attic"],
               s=0.4, alpha=0.7, linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    ax.set_facecolor(BG)
    ax.set_title(title, color="white", fontsize=7.5, pad=3)
    ax.tick_params(colors="#444466", labelsize=5)
    for sp in ax.spines.values(): sp.set_edgecolor("#222244")

# Row 1: orientations
for i, (label, R) in enumerate(ORIENT.items()):
    ax = axes[0, i]
    rotated = (R @ attic_raw.T).T
    T_test = T_attic.copy()
    attic_global = apply4(rotated, T_test)
    draw_stack(ax, attic_global, label)

for i in range(len(ORIENT), ncols):
    axes[0, i].set_visible(False)

# Row 2: X shifts (using current orientation)
for i, dx in enumerate(X_SHIFTS):
    ax = axes[1, i]
    T_test = T_attic.copy()
    T_test[0, 3] += dx
    attic_global = apply4(attic_raw, T_test)
    draw_stack(ax, attic_global, f"X +{dx}m")

for i in range(len(X_SHIFTS), ncols):
    axes[1, i].set_visible(False)

out_img = OUT / "04e_attic_tune.png"
plt.tight_layout()
plt.savefig(out_img, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
subprocess.run(["open", str(out_img)])
print(f"Saved → {out_img}")
