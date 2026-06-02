# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "trimesh", "matplotlib", "scipy"]
# ///

"""
Step 2 — Coordinate system normalization.

Polycam photogrammetry GLBs use GLTF convention (Y-up, right-handed).
LiDAR PLYs use Z-up. Convert all photogrammetry to Z-up so axes match
before registration, then render side-by-side for visual confirmation.

GLTF Y-up → Z-up:  rotate +90° about X
  new_x =  old_x
  new_y = -old_z
  new_z =  old_y   (height stays positive)
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import subprocess
from config import FLOOR_ORDER, OUT

# Rotation +90° about X: GLTF Y-up → Z-up
GLTF_TO_ZUP = np.array([
    [1,  0,  0],
    [0,  0, -1],
    [0,  1,  0],
], dtype=np.float32)

# Additional per-capture flip needed for main_floor and top_floor photogrammetry
FLIP_XY = np.diag([-1., -1., 1.]).astype(np.float32)

PER_FLOOR_EXTRA = {
    "main_floor": FLIP_XY,
    "top_floor":  FLIP_XY,
}


def apply(pts: np.ndarray, R: np.ndarray) -> np.ndarray:
    return (R @ pts.T).T


# ── Load cached point clouds from step 1 ─────────────────────────────────────
cache_in = OUT / "01_pointclouds.pkl"
print(f"Loading cache: {cache_in}")
with open(cache_in, "rb") as f:
    data = pickle.load(f)

photo_raw   = data["photo"]
lidar_pts   = data["lidar"]
photo_norm  = {}

print("\nApplying GLTF→Z-up transform to all photogrammetry clouds...")
for name, pts in photo_raw.items():
    normed = apply(pts, GLTF_TO_ZUP)
    if name in PER_FLOOR_EXTRA:
        normed = apply(normed, PER_FLOOR_EXTRA[name])
        extra = " + flip X+Y"
    else:
        extra = ""
    photo_norm[name] = normed
    bb_before = pts.max(0) - pts.min(0)
    bb_after  = normed.max(0) - normed.min(0)
    print(f"  {name}{extra}: bbox {bb_before[0]:.1f}×{bb_before[1]:.1f}×{bb_before[2]:.1f}"
          f" → {bb_after[0]:.1f}×{bb_after[1]:.1f}×{bb_after[2]:.1f}")

# ── Render comparison: photo (normalized) vs LiDAR ───────────────────────────
FLOOR_COL = {"basement": "#e8b86d", "main_floor": "#7ec8e3",
             "top_floor": "#b5ead7", "attic": "#c9b1ff"}
STAIR_COL = {"main_to_top": "#ff9980", "main_to_basement": "#ffcc80"}
BG        = "#050510"
S         = 4

all_floors = FLOOR_ORDER + ["attic"]
n_rows     = len(all_floors) + 1  # +1 for staircases

fig = plt.figure(figsize=(18, n_rows * 3.5 + 1), facecolor="#0d0d1a")
fig.suptitle("Step 2 — Coord Normalisation: Photo (Y-up→Z-up) vs LiDAR",
             color="white", fontsize=13, fontweight="bold", y=0.995)
gs = gridspec.GridSpec(n_rows, 4, figure=fig, hspace=0.45, wspace=0.2,
                       top=0.975, bottom=0.01, left=0.03, right=0.97)


def panel(ax, pts, title, color, view="xy", s=0.3):
    xs = pts[::S, 0]
    ys = pts[::S, 1] if view == "xy" else pts[::S, 2]
    ax.scatter(xs, ys, c=color, s=s, alpha=0.65, linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    ax.set_facecolor(BG)
    ax.set_title(title, color="white", fontsize=7.5, pad=3)
    ax.tick_params(colors="#444466", labelsize=6)
    for sp in ax.spines.values(): sp.set_edgecolor("#222244")


for row, floor in enumerate(all_floors):
    col = FLOOR_COL.get(floor, "#ffffff")

    if floor in photo_norm:
        p = photo_norm[floor]
        panel(fig.add_subplot(gs[row, 0]), p, f"{floor} photo — top (XY)", col)
        panel(fig.add_subplot(gs[row, 1]), p, f"{floor} photo — side (XZ)", col, "xz")

    if floor in lidar_pts:
        l = lidar_pts[floor]
        panel(fig.add_subplot(gs[row, 2]), l, f"{floor} LiDAR — top (XY)", col, s=0.2)
        panel(fig.add_subplot(gs[row, 3]), l, f"{floor} LiDAR — side (XZ)", col, "xz", s=0.2)

# Staircase row
stair_row = len(all_floors)
for ci, (name, pts) in enumerate(
        {k: photo_norm[k] for k in ["main_to_top", "main_to_basement"]
         if k in photo_norm}.items()):
    col = STAIR_COL.get(name, "#fff")
    panel(fig.add_subplot(gs[stair_row, ci * 2]),     pts, f"{name} — top",  col)
    panel(fig.add_subplot(gs[stair_row, ci * 2 + 1]), pts, f"{name} — side", col, "xz")

out_img = OUT / "02_coord_normalized.png"
plt.savefig(out_img, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nSaved → {out_img}")
subprocess.run(["open", str(out_img)])

# Cache normalized coords
cache_out = OUT / "02_pointclouds_normalized.pkl"
with open(cache_out, "wb") as f:
    pickle.dump({"photo": photo_norm, "lidar": lidar_pts}, f)
print(f"Cache → {cache_out}")
