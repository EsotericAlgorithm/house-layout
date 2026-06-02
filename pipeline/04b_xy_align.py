# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "matplotlib", "scipy", "trimesh"]
# ///

"""
Step 4b — XY horizontal alignment.

After Z stacking, each floor's photogrammetry is still in its own local XY.
We find the rightmost dense wall feature per floor (the exterior wall) and
shift all upper floors to align their right edge with the basement.
Attic is excluded from this alignment (user: align on other features later).
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


def rightmost_wall_x(pts, bin_size=0.05):
    """Find the X position of the rightmost dense vertical feature."""
    x = pts[:, 0]
    x_max = x.max()
    # Look in the rightmost 30% of the X range
    search_lo = x_max - (x_max - x.min()) * 0.30
    x_right = x[x >= search_lo]
    bins = np.arange(search_lo, x_max + bin_size, bin_size)
    if len(bins) < 2:
        return float(x_max)
    counts, edges = np.histogram(x_right, bins=bins)
    # Take the rightmost peak (highest X with significant density)
    threshold = counts.max() * 0.25
    candidates = np.where(counts >= threshold)[0]
    if len(candidates) == 0:
        return float(x_max)
    return float(edges[candidates[-1]] + bin_size / 2)


cache = OUT / "04_global_transforms.pkl"
with open(cache, "rb") as f:
    data = pickle.load(f)

global_T      = data["global_T"]
photo_pts     = data["photo_pts"]
lidar_clipped = data["lidar_clipped"]
per_floor_T   = data["per_floor_T"]


def apply4(pts, T):
    ones = np.ones((len(pts), 1), dtype=np.float64)
    return (T @ np.hstack([pts, ones]).T).T[:, :3]


print("Finding rightmost wall X per floor (in global Z space)...")
right_x = {}
for floor in ["basement", "main_floor", "top_floor"]:
    pts = apply4(photo_pts[floor].astype(np.float64), global_T[floor])
    rx  = rightmost_wall_x(pts)
    right_x[floor] = rx
    print(f"  {floor}: rightmost wall X = {rx:.3f}m")

# Basement is reference — shift others to match
ref_x = right_x["basement"]
print(f"\nBasement right wall (reference): X = {ref_x:.3f}m")

xy_shifts = {"basement": 0.0, "attic": 0.0}
for floor in ["main_floor", "top_floor"]:
    dx = ref_x - right_x[floor]
    xy_shifts[floor] = dx
    print(f"  {floor}: shift X by {dx:+.3f}m")

# Bake X shifts into global_T
for floor, dx in xy_shifts.items():
    global_T[floor][0, 3] += dx

# ── Re-render XZ side view ────────────────────────────────────────────────────
FLOOR_COLORS = {"basement": "#e8b86d", "main_floor": "#7ec8e3",
                "top_floor": "#b5ead7", "attic": "#c9b1ff"}
BG = "#050510"; S = 6

fig, axes = plt.subplots(1, 2, figsize=(20, 8), facecolor="#0d0d1a")
fig.suptitle("Step 4b — XY Aligned Global Stack",
             color="white", fontsize=13, fontweight="bold")

for ax, (xi, yi, label) in zip(axes, [(0, 1, "Top-down (XY)"), (0, 2, "Side (XZ)")]):
    ax.set_facecolor(BG)
    ax.set_title(label, color="white", fontsize=10)
    ax.tick_params(colors="#444466", labelsize=6)
    for sp in ax.spines.values(): sp.set_edgecolor("#222244")

    for floor, T in global_T.items():
        pts = apply4(photo_pts[floor].astype(np.float64), T)
        col = FLOOR_COLORS[floor]
        ax.scatter(pts[::S, xi], pts[::S, yi],
                   c=col, s=0.3, alpha=0.6, linewidths=0,
                   rasterized=True, label=floor)
        if xi == 0 and yi == 2 and floor != "attic":
            ax.axvline(pts[:, 0].max(), color=col,
                       linewidth=0.8, alpha=0.4, linestyle="--")

    ax.set_aspect("equal")
    ax.legend(fontsize=8, facecolor="#111122", labelcolor="white",
              edgecolor="#333355", markerscale=8)

out_img = OUT / "04b_xy_aligned.png"
plt.tight_layout()
plt.savefig(out_img, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nSaved → {out_img}")
subprocess.run(["open", str(out_img)])

# Save updated transforms
cache_out = OUT / "04b_global_transforms.pkl"
with open(cache_out, "wb") as f:
    pickle.dump({"global_T": global_T,
                 "photo_pts": photo_pts,
                 "lidar_clipped": lidar_clipped,
                 "per_floor_T": per_floor_T}, f)
print(f"Cache → {cache_out}")
