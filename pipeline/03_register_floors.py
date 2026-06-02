# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "trimesh", "matplotlib", "scipy"]
# ///

"""
Step 3 — Per-floor LiDAR → Photogrammetry registration.

For each floor:
  1. Coarse align: match centroids + scale LiDAR to photo bounding box
  2. ICP refinement (trimesh, point-to-point)
  3. Store 4×4 transform matrix

Output: transforms dict saved to output/03_transforms.pkl
        overlay render for visual QC
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import numpy as np
import pickle
import trimesh
import trimesh.registration
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import subprocess
from config import OUT

ICP_MAX_ITER  = 100
ICP_THRESHOLD = 1e-5
# Subsample to this many points for ICP (speed vs accuracy)
ICP_SUBSAMPLE = 20_000
# Extra margin (metres) beyond photogrammetry bbox when clipping LiDAR for ICP
BBOX_MARGIN = 0.5

FLOORS = ["basement", "main_floor", "top_floor"]

cache = OUT / "02_pointclouds_normalized.pkl"
print(f"Loading: {cache}")
with open(cache, "rb") as f:
    data = pickle.load(f)

photo_pts = data["photo"]
lidar_pts = data["lidar"]


def coarse_align(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """Translate src centroid → tgt centroid. Returns 4×4 matrix."""
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = tgt.mean(0) - src.mean(0)
    return T


def apply_transform(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    ones = np.ones((len(pts), 1), dtype=np.float64)
    return (T @ np.hstack([pts, ones]).T).T[:, :3]


def clip_to_bbox(pts: np.ndarray, ref: np.ndarray, margin: float) -> np.ndarray:
    """Keep only pts that fall within ref's XY bounding box + margin."""
    lo = ref.min(0)[:2] - margin
    hi = ref.max(0)[:2] + margin
    mask = np.all((pts[:, :2] >= lo) & (pts[:, :2] <= hi), axis=1)
    return pts[mask]


def icp(src: np.ndarray, tgt: np.ndarray, init: np.ndarray) -> tuple[np.ndarray, float]:
    """Thin wrapper around trimesh ICP. Returns (4×4 transform, final cost)."""
    src_s = src[np.random.default_rng(0).choice(len(src), min(ICP_SUBSAMPLE, len(src)), replace=False)]
    tgt_s = tgt[np.random.default_rng(1).choice(len(tgt), min(ICP_SUBSAMPLE, len(tgt)), replace=False)]
    matrix, _, cost = trimesh.registration.icp(
        src_s.astype(np.float64),
        tgt_s.astype(np.float64),
        initial=init,
        threshold=ICP_THRESHOLD,
        max_iterations=ICP_MAX_ITER,
    )
    return matrix, cost


transforms   = {}
lidar_clipped_all = {}  # clipped clouds used for ICP — what we'll carry forward
BG = "#050510"
S  = 6

n_floors = len(FLOORS)
fig = plt.figure(figsize=(15, n_floors * 4 + 1), facecolor="#0d0d1a")
fig.suptitle("Step 3 — LiDAR → Photo Registration (overlay QC)",
             color="white", fontsize=13, fontweight="bold", y=0.99)
gs = gridspec.GridSpec(n_floors, 3, figure=fig, hspace=0.4, wspace=0.2,
                       top=0.96, bottom=0.02, left=0.04, right=0.96)

FLOOR_COL = {"basement": "#e8b86d", "main_floor": "#7ec8e3", "top_floor": "#b5ead7"}

for row, floor in enumerate(FLOORS):
    print(f"\n── {floor} ──────────────────────────────")
    photo = photo_pts[floor].astype(np.float64)
    lidar = lidar_pts[floor].astype(np.float64)
    col   = FLOOR_COL[floor]

    # 1. Coarse align (using bbox-clipped lidar so outliers don't skew centroid)
    lidar_clipped = clip_to_bbox(lidar, photo, BBOX_MARGIN)
    clipped_pct = 100 * len(lidar_clipped) / len(lidar)
    print(f"  BBox clip: {len(lidar):,} → {len(lidar_clipped):,} pts ({clipped_pct:.0f}%)")
    T_coarse = coarse_align(lidar_clipped, photo)
    lidar_coarse = apply_transform(lidar_clipped, T_coarse)
    print(f"  Coarse: lidar centroid {lidar_clipped.mean(0).round(2)} → {photo.mean(0).round(2)}")

    # 2. ICP on clipped cloud, then apply resulting transform to full cloud
    print(f"  Running ICP (max {ICP_MAX_ITER} iter, {ICP_SUBSAMPLE:,} pts each)...")
    T_icp, cost = icp(lidar_coarse, photo, np.eye(4))
    T_final = T_icp @ T_coarse
    transforms[floor] = T_final
    print(f"  ICP cost: {cost:.6f}")

    # Use clipped cloud for render + downstream merge — drop the out-of-bounds tail
    lidar_registered = apply_transform(lidar_clipped, T_final)
    lidar_clipped_all[floor] = lidar_clipped

    # 3. QC panels
    def panel(ax, pts_a, pts_b, title, view="xy"):
        xi, yi = (0, 1) if view == "xy" else (0, 2)
        ax.scatter(pts_a[::S, xi], pts_a[::S, yi],
                   c="#ffffff", s=0.2, alpha=0.35, linewidths=0, rasterized=True,
                   label="photo")
        ax.scatter(pts_b[::S, xi], pts_b[::S, yi],
                   c=col, s=0.2, alpha=0.5, linewidths=0, rasterized=True,
                   label="lidar reg.")
        ax.set_aspect("equal")
        ax.set_facecolor(BG)
        ax.set_title(title, color="white", fontsize=8, pad=3)
        ax.tick_params(colors="#444466", labelsize=6)
        for sp in ax.spines.values(): sp.set_edgecolor("#222244")

    panel(fig.add_subplot(gs[row, 0]), photo, lidar_registered,
          f"{floor} — top-down overlay")
    panel(fig.add_subplot(gs[row, 1]), photo, lidar_registered,
          f"{floor} — side overlay", view="xz")

    # Cost label panel
    ax_info = fig.add_subplot(gs[row, 2])
    ax_info.set_facecolor(BG)
    ax_info.axis("off")
    info = (f"{floor}\n\n"
            f"LiDAR pts: {len(lidar):,}\n"
            f"Photo pts:  {len(photo):,}\n\n"
            f"ICP cost:   {cost:.5f}\n\n"
            f"Translation (m):\n"
            f"  x={T_final[0,3]:.3f}\n"
            f"  y={T_final[1,3]:.3f}\n"
            f"  z={T_final[2,3]:.3f}")
    ax_info.text(0.05, 0.95, info, transform=ax_info.transAxes,
                 color="white", fontsize=8, va="top", fontfamily="monospace")
    ax_info.set_title(f"{floor} — registration stats", color="white", fontsize=8, pad=3)
    for sp in ax_info.spines.values(): sp.set_edgecolor("#222244")

out_img = OUT / "03_registration.png"
plt.savefig(out_img, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nSaved → {out_img}")
subprocess.run(["open", str(out_img)])

# Save transforms
cache_out = OUT / "03_transforms.pkl"
with open(cache_out, "wb") as f:
    pickle.dump({"transforms": transforms,
                 "photo": photo_pts,
                 "lidar": lidar_pts,
                 "lidar_clipped": lidar_clipped_all}, f)
print(f"Transforms → {cache_out}")
print("\nDone. Transforms saved for all floors:")
for floor, T in transforms.items():
    print(f"  {floor}: t=({T[0,3]:.3f}, {T[1,3]:.3f}, {T[2,3]:.3f})")
