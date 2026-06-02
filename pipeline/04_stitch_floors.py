# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "trimesh", "matplotlib", "scipy"]
# ///

"""
Step 4 — Vertical floor stitching.

Each staircase photogrammetry capture spans two adjacent floors.
Strategy:
  1. Each floor's photogrammetry is already in its own local space.
  2. Register each staircase capture to the floor below it (ICP).
  3. Use that transform to express the floor above in the same space.
  4. Chain: basement → main_floor → top_floor → attic
     with staircase captures as the connective tissue.

Output: a global Z offset per floor that stacks them correctly,
        plus a merged render of all floors in one coordinate space.
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
import subprocess
from config import PHOTO_STAIRS, PHOTO_FLOORS, OUT

ICP_SUBSAMPLE = 15_000
ICP_MAX_ITER  = 80

cache = OUT / "03_transforms.pkl"
with open(cache, "rb") as f:
    data = pickle.load(f)

photo_pts  = data["photo"]   # normalized per-floor photo clouds (local space)
transforms = data["transforms"]
lidar_clipped = data["lidar_clipped"]

FLOOR_COLORS = {
    "basement":   "#e8b86d",
    "main_floor": "#7ec8e3",
    "top_floor":  "#b5ead7",
    "attic":      "#c9b1ff",
}

# Coord normalization applied in step 2
GLTF_TO_ZUP = np.array([[1,0,0],[0,0,-1],[0,1,0]], dtype=np.float32)
FLIP_XY     = np.diag([-1.,-1.,1.]).astype(np.float32)
PER_FLOOR_EXTRA = {"main_floor": FLIP_XY, "top_floor": FLIP_XY}


def normalize(pts, name):
    pts = (GLTF_TO_ZUP @ pts.T).T
    if name in PER_FLOOR_EXTRA:
        pts = (PER_FLOOR_EXTRA[name] @ pts.T).T
    return pts


def load_glb(path, n=60_000):
    scene = trimesh.load(str(path), force="scene")
    meshes = [g for g in scene.geometry.values()
              if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0]
    mesh = trimesh.util.concatenate(meshes)
    pts, _ = trimesh.sample.sample_surface(mesh, count=n)
    return pts.astype(np.float32)


def apply4(pts, T):
    ones = np.ones((len(pts), 1), dtype=np.float64)
    return (T @ np.hstack([pts, ones]).T).T[:, :3]


def run_icp(src, tgt):
    rng = np.random.default_rng(42)
    src_s = src[rng.choice(len(src), min(ICP_SUBSAMPLE, len(src)), replace=False)]
    tgt_s = tgt[rng.choice(len(tgt), min(ICP_SUBSAMPLE, len(tgt)), replace=False)]
    # coarse: centroid align
    T_c = np.eye(4); T_c[:3,3] = tgt_s.mean(0) - src_s.mean(0)
    src_c = apply4(src_s, T_c)
    T_icp, _, cost = trimesh.registration.icp(
        src_c.astype(np.float64), tgt_s.astype(np.float64),
        threshold=1e-5, max_iterations=ICP_MAX_ITER)
    return T_icp @ T_c, cost


def find_floor_plane_z(pts, bin_size=0.05, window=0.3):
    """Find the dominant floor surface Z by looking for the lowest dense
    horizontal band in the point cloud."""
    z = pts[:, 2]
    z_min, z_max = z.min(), z.max()
    bins = np.arange(z_min, z_max + bin_size, bin_size)
    counts, edges = np.histogram(z, bins=bins)
    # Find densest bin in the lower 40% of Z range
    cutoff = z_min + (z_max - z_min) * 0.4
    lower_mask = edges[:-1] < cutoff
    lower_counts = counts * lower_mask
    best_bin = np.argmax(lower_counts)
    return float(edges[best_bin] + bin_size / 2)


# ── Floor plane extraction ────────────────────────────────────────────────────
def find_floor_plane_z(pts, bin_size=0.05):
    """Find dominant floor Z — lowest dense horizontal band."""
    z = pts[:, 2]
    lo = z.min(); hi = z.min() + (z.max() - z.min()) * 0.4
    mask = (z >= lo) & (z <= hi)
    bins = np.arange(lo, hi + bin_size, bin_size)
    if len(bins) < 2 or mask.sum() < 10:
        return float(z.min())
    counts, edges = np.histogram(z[mask], bins=bins)
    return float(edges[np.argmax(counts)] + bin_size / 2)


def find_landing_zs(stair_pts, bin_size=0.06, min_gap=0.8):
    """Find the two floor-landing Z levels inside a staircase capture.
    Returns (lower_z, upper_z) — the two dominant horizontal bands."""
    z = stair_pts[:, 2]
    bins = np.arange(z.min(), z.max() + bin_size, bin_size)
    counts, edges = np.histogram(z, bins=bins)
    mid = edges[:-1] + bin_size / 2

    # Find all local maxima
    from scipy.signal import find_peaks
    peaks, props = find_peaks(counts, height=counts.max() * 0.15,
                               distance=int(min_gap / bin_size))
    if len(peaks) < 2:
        # Fallback: top and bottom quartile peaks
        lo_idx = np.argmax(counts[:len(counts)//2])
        hi_idx = len(counts)//2 + np.argmax(counts[len(counts)//2:])
        return float(mid[lo_idx]), float(mid[hi_idx])

    # Take the two strongest peaks
    peak_heights = counts[peaks]
    top2 = peaks[np.argsort(peak_heights)[-2:]]
    top2_sorted = sorted(top2)
    return float(mid[top2_sorted[0]]), float(mid[top2_sorted[1]])


# ── Load staircase GLBs and find their landing planes ────────────────────────
print("Loading staircase GLBs and detecting landing Z levels...")
stair_landings = {}
for name, path in PHOTO_STAIRS.items():
    pts = load_glb(path, n=60_000)
    pts = normalize(pts, name)
    lo_z, hi_z = find_landing_zs(pts)
    stair_landings[name] = (lo_z, hi_z)
    span = hi_z - lo_z
    print(f"  {name}: lower landing={lo_z:.3f}m  upper landing={hi_z:.3f}m  "
          f"floor-to-floor={span:.3f}m")

h_basement_to_main = stair_landings["main_to_basement"][1] - stair_landings["main_to_basement"][0]
h_main_to_top      = stair_landings["main_to_top"][1]      - stair_landings["main_to_top"][0]
print(f"\n  Derived story heights:")
print(f"    basement → main_floor : {h_basement_to_main:.3f}m")
print(f"    main_floor → top_floor: {h_main_to_top:.3f}m")

# ── Find floor plane in each registered LiDAR ────────────────────────────────
print("\nExtracting floor planes from registered LiDAR...")
floor_z = {}
for floor, lidar in lidar_clipped.items():
    T    = transforms[floor]
    ones = np.ones((len(lidar), 1))
    reg  = (T @ np.hstack([lidar, ones]).T).T[:, :3]
    fz   = find_floor_plane_z(reg)
    floor_z[floor] = fz
    print(f"  {floor}: local floor plane Z={fz:.3f}m")

# ── Stack floors: basement floor = Z 0, chain upward ─────────────────────────
print("\nComputing global Z stack...")
global_T = {}
target_z = 0.0   # basement floor at global Z=0

# h_to_next = story height from THIS floor up to the NEXT floor
for floor, h_to_next in [("basement",   h_basement_to_main),
                          ("main_floor", h_main_to_top),
                          ("top_floor",  None)]:
    shift = target_z - floor_z[floor]
    T = np.eye(4); T[2, 3] = shift
    global_T[floor] = T
    print(f"  {floor}: shift={shift:+.3f}m  floor→Z={target_z:.3f}m")
    if h_to_next is not None:
        target_z += h_to_next

# Attic: no LiDAR, place one main→top story height above top floor
target_z += h_main_to_top
T_attic = np.eye(4); T_attic[2, 3] = target_z
global_T["attic"] = T_attic
print(f"  attic: estimated floor Z={target_z:.3f}m  "
      f"(top_floor + {h_main_to_top:.3f}m estimated)")

# ── Render global stack ───────────────────────────────────────────────────────
print("\nRendering global floor stack...")
BG = "#050510"; S = 6
fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor="#0d0d1a")
fig.suptitle("Step 4 — Global Floor Stack (photo clouds)",
             color="white", fontsize=13, fontweight="bold")

for ax, (xi, yi, label) in zip(axes, [(0,1,"Top-down (XY)"), (0,2,"Side (XZ)")]):
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
        # Mark right edge (max X) for XZ view so we can assess horizontal alignment
        if xi == 0 and yi == 2:
            x_max = pts[:, 0].max()
            ax.axvline(x_max, color=col, linewidth=0.8, alpha=0.5, linestyle="--")

    ax.set_aspect("equal")
    ax.legend(fontsize=8, facecolor="#111122", labelcolor="white",
              edgecolor="#333355", markerscale=8)

out_img = OUT / "04_global_stack.png"
plt.tight_layout()
plt.savefig(out_img, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {out_img}")
subprocess.run(["open", str(out_img)])

# Save
cache_out = OUT / "04_global_transforms.pkl"
with open(cache_out, "wb") as f:
    pickle.dump({"global_T": global_T,
                 "photo_pts": photo_pts,
                 "lidar_clipped": lidar_clipped,
                 "per_floor_T": transforms}, f)
print(f"Cache → {cache_out}")

print("\nGlobal Z offsets per floor:")
for floor, T in global_T.items():
    print(f"  {floor}: Z={T[2,3]:.3f}m")
