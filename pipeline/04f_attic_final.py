# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "matplotlib", "trimesh"]
# ///

"""Applies final attic correction: +6m X, Z raised so attic floor = top floor ceiling."""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import subprocess
from config import OUT

ATTIC_X_SHIFT = 6.0

cache = OUT / "04d_final_transforms.pkl"
with open(cache, "rb") as f:
    data = pickle.load(f)

global_T      = data["global_T"]
photo_pts     = data["photo_pts"]
lidar_clipped = data["lidar_clipped"]
per_floor_T   = data["per_floor_T"]

def apply4(pts, T):
    ones = np.ones((len(pts), 1), dtype=np.float64)
    return (T @ np.hstack([pts, ones]).T).T[:, :3]

def find_floor_z(pts, bin_size=0.05):
    z = pts[:, 2]
    lo, hi = z.min(), z.min() + (z.max() - z.min()) * 0.35
    bins = np.arange(lo, hi + bin_size, bin_size)
    if len(bins) < 2: return float(lo)
    counts, edges = np.histogram(z[(z >= lo) & (z <= hi)], bins=bins)
    return float(edges[np.argmax(counts)] + bin_size / 2)

def find_ceiling_z(pts, bin_size=0.05):
    z = pts[:, 2]
    lo = z.min() + (z.max() - z.min()) * 0.65
    bins = np.arange(lo, z.max() + bin_size, bin_size)
    if len(bins) < 2: return float(z.max())
    counts, edges = np.histogram(z[z >= lo], bins=bins)
    return float(edges[np.argmax(counts)] + bin_size / 2)

# Find top floor ceiling in global space
top_global = apply4(photo_pts["top_floor"].astype(np.float64), global_T["top_floor"])
top_ceiling_z = find_ceiling_z(top_global)
print(f"Top floor ceiling in global space: Z = {top_ceiling_z:.3f}m")

# Find attic floor plane in its local space (before global transform)
attic_local = photo_pts["attic"].astype(np.float64)
attic_local_transformed = apply4(attic_local, global_T["attic"])
attic_floor_z_current = find_floor_z(attic_local_transformed)
print(f"Attic floor (current global): Z = {attic_floor_z_current:.3f}m")

# Compute Z correction so attic floor sits at top floor ceiling
dz = top_ceiling_z - attic_floor_z_current
print(f"Attic Z correction: {dz:+.3f}m")

# Apply corrections
global_T["attic"][0, 3] += ATTIC_X_SHIFT
global_T["attic"][2, 3] += dz

print(f"\nFinal attic transform: X={global_T['attic'][0,3]:.3f}m  Z={global_T['attic'][2,3]:.3f}m")
print("\nAll global offsets:")
for floor, T in global_T.items():
    print(f"  {floor}: X={T[0,3]:.3f}m  Z={T[2,3]:.3f}m")

# ── Render ────────────────────────────────────────────────────────────────────
FLOOR_COLORS = {"basement": "#e8b86d", "main_floor": "#7ec8e3",
                "top_floor": "#b5ead7", "attic": "#c9b1ff"}
BG = "#050510"; S = 6

fig, axes = plt.subplots(1, 2, figsize=(20, 9), facecolor="#0d0d1a")
fig.suptitle("Step 4f — Final Global Stack (attic corrected)",
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

out_img = OUT / "04f_final_stack.png"
plt.tight_layout()
plt.savefig(out_img, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
subprocess.run(["open", str(out_img)])
print(f"\nSaved → {out_img}")

cache_out = OUT / "04f_final_transforms.pkl"
with open(cache_out, "wb") as f:
    pickle.dump({"global_T": global_T, "photo_pts": photo_pts,
                 "lidar_clipped": lidar_clipped, "per_floor_T": per_floor_T}, f)
print(f"Cache → {cache_out}")
