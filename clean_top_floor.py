# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "matplotlib", "scipy"]
# ///

"""
Cleans the top floor scan:
1. RANSAC plane detection in artifact zone — removes the ~45° diagonal ghost plane
2. Statistical outlier removal (SOR) globally
3. Saves cleaned PLY and renders before/after comparison

The artifact is a ghost scan plane whose XY normal is ~45° (π/4) from the room
axes — i.e. |nx| ≈ |ny| with normal_z ≈ 0. Standard wall planes run parallel to
room axes (|nx|>>|ny| or vice-versa), so the XY normal angle distinguishes them.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.spatial import cKDTree
import os
import subprocess

PLY_IN  = os.path.join(os.path.dirname(__file__), "docs", "top_floor", "5_29_2026.ply")
PLY_OUT = os.path.join(os.path.dirname(__file__), "docs", "top_floor", "5_29_2026_clean.ply")
IMG_OUT = os.path.join(os.path.dirname(__file__), "top_floor_cleaned.png")

# Artifact zone bounds (upper-right corner)
ART_X_MIN, ART_Y_MIN = 1.0, 1.0

# RANSAC params
RANSAC_ITERS      = 600
PLANE_INLIER_DIST = 0.04   # metres
MIN_PLANE_POINTS  = 300

# Diagonal plane detection:
#   wall-like: |normal_z| < this
WALL_Z_THRESH = 0.15
#   diagonal in XY: ratio |nx|/|ny| between these bounds (1.0 = perfect 45°)
DIAG_RATIO_LO = 0.35   # tan(~20°) — anything more diagonal than this
DIAG_RATIO_HI = 2.85   # tan(~70°) — and less axis-aligned than this

# SOR params
SOR_K   = 20
SOR_STD = 2.0


# ── helpers ──────────────────────────────────────────────────────────────────

def parse_ply(path):
    props = []
    n_verts = 0
    with open(path, "rb") as f:
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            if line == "end_header":
                break
            if line.startswith("element vertex"):
                n_verts = int(line.split()[-1])
            if line.startswith("property"):
                parts = line.split()
                props.append((parts[1], parts[2]))
        type_map = {"double": "f8", "float": "f4", "uchar": "u1", "uint8": "u1",
                    "int": "i4", "uint": "u4", "short": "i2", "ushort": "u2"}
        dt = np.dtype([(name, type_map[t]) for t, name in props])
        data = np.frombuffer(f.read(n_verts * dt.itemsize), dtype=dt)
    return data, dt


def write_ply(path, data):
    type_str = {"float32": "float", "float64": "double",
                "uint8": "uchar", "int32": "int", "uint32": "uint"}
    with open(path, "wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\ncomment Cleaned by layout pipeline\n")
        f.write(f"element vertex {len(data)}\n".encode())
        for name in data.dtype.names:
            dt_name = type_str.get(str(data.dtype[name]), str(data.dtype[name]))
            f.write(f"property {dt_name} {name}\n".encode())
        f.write(b"end_header\n")
        f.write(data.tobytes())


def fit_plane_ransac(pts, n_iters, inlier_dist, rng):
    best_inliers = np.zeros(len(pts), dtype=bool)
    best_normal = np.array([0., 0., 1.])
    for _ in range(n_iters):
        idx = rng.choice(len(pts), 3, replace=False)
        p0, p1, p2 = pts[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        nlen = np.linalg.norm(normal)
        if nlen < 1e-10:
            continue
        normal /= nlen
        dist = np.abs(pts @ normal - normal @ p0)
        inliers = dist < inlier_dist
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_normal = normal
    return best_normal, best_inliers


def is_diagonal_wall(normal):
    """True if the plane is wall-like AND has a ~45° XY normal (ghost plane)."""
    nz = abs(normal[2])
    if nz >= WALL_Z_THRESH:
        return False  # floor or ceiling, not a wall
    nx, ny = abs(normal[0]), abs(normal[1])
    if ny < 1e-9:
        return False
    ratio = nx / ny
    return DIAG_RATIO_LO <= ratio <= DIAG_RATIO_HI


def statistical_outlier_removal(pts, k, std_mult):
    print(f"  SOR: KD-tree over {len(pts):,} points...")
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=k + 1)
    mean_dists = dists[:, 1:].mean(axis=1)
    threshold = mean_dists.mean() + std_mult * mean_dists.std()
    return mean_dists <= threshold


# ── main ─────────────────────────────────────────────────────────────────────

print("Loading top floor...")
data, dt = parse_ply(PLY_IN)
x = data["x"].astype(np.float64)
y = data["y"].astype(np.float64)
z = data["z"].astype(np.float64)
pts = np.stack([x, y, z], axis=1)
print(f"  {len(pts):,} points loaded")

# ── Step 1: RANSAC diagonal wall removal in artifact zone ─────────────────────
print("\nStep 1: RANSAC in artifact zone (targeting ~45° diagonal wall planes)...")
art_mask = (x >= ART_X_MIN) & (y >= ART_Y_MIN)
art_idx  = np.where(art_mask)[0]
art_pts  = pts[art_idx]
print(f"  Artifact zone: {len(art_pts):,} points")

keep_mask  = np.ones(len(pts), dtype=bool)
remaining  = np.ones(len(art_pts), dtype=bool)
rng        = np.random.default_rng(42)
diag_removed = 0
plane_n = 0

while remaining.sum() >= MIN_PLANE_POINTS:
    work_pts = art_pts[remaining]
    normal, inliers_local = fit_plane_ransac(work_pts, RANSAC_ITERS, PLANE_INLIER_DIST, rng)
    if inliers_local.sum() < MIN_PLANE_POINTS:
        break

    nz    = abs(normal[2])
    nx_ny = abs(normal[0]) / max(abs(normal[1]), 1e-9)
    diag  = is_diagonal_wall(normal)
    xy_deg = np.degrees(np.arctan2(abs(normal[1]), abs(normal[0])))
    label = "DIAGONAL GHOST (removing)" if diag else (
            "FLOOR/CEILING" if nz >= WALL_Z_THRESH else f"WALL (axis-aligned, XY={xy_deg:.0f}°)")
    print(f"  Plane {plane_n+1}: {inliers_local.sum():,} pts  "
          f"nz={nz:.3f}  |nx/ny|={nx_ny:.2f}  XY={xy_deg:.0f}°  → {label}")

    if diag:
        global_idx = art_idx[np.where(remaining)[0][inliers_local]]
        keep_mask[global_idx] = False
        diag_removed += inliers_local.sum()

    remaining_idx = np.where(remaining)[0]
    remaining[remaining_idx[inliers_local]] = False
    plane_n += 1

print(f"  Removed {diag_removed:,} diagonal ghost points")

# ── Step 2: Statistical outlier removal ──────────────────────────────────────
print("\nStep 2: Statistical outlier removal (global)...")
step1_pts = pts[keep_mask]
sor_keep  = statistical_outlier_removal(step1_pts, SOR_K, SOR_STD)
print(f"  SOR removed {(~sor_keep).sum():,} outliers")

final_keep_global = np.where(keep_mask)[0][sor_keep]
final_mask = np.zeros(len(pts), dtype=bool)
final_mask[final_keep_global] = True
print(f"\nFinal: {len(pts):,} → {final_mask.sum():,} pts  (removed {(~final_mask).sum():,} total)")

# ── Save ─────────────────────────────────────────────────────────────────────
print(f"\nSaving → {PLY_OUT}")
write_ply(PLY_OUT, data[final_mask])
print("Saved.")

# ── Before/after render ──────────────────────────────────────────────────────
print("\nRendering before/after...")
S      = 6
norm_c = Normalize(vmin=z.min(), vmax=z.max())
cmap   = plt.cm.plasma
BG     = "#050510"

fig = plt.figure(figsize=(22, 14), facecolor="#0d0d1a")
fig.suptitle("Top Floor — Before / After Cleanup", fontsize=14,
             color="white", fontweight="bold", y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.2,
                       top=0.93, bottom=0.05, left=0.04, right=0.93)

xf = pts[final_mask, 0]
yf = pts[final_mask, 1]
zf = pts[final_mask, 2]

def panel(pos, xs, ys, zs, title, border="#222244"):
    ax = fig.add_subplot(pos, facecolor=BG)
    ax.scatter(xs[::S], ys[::S], c=cmap(norm_c(zs[::S])),
               s=0.4, alpha=0.7, linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    ax.set_title(title, color="white", fontsize=10, pad=5)
    ax.tick_params(colors="#555555", labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor(border)
    return ax

panel(gs[0, 0], x,  y,  z,  f"BEFORE — top-down\n{len(x):,} pts")
panel(gs[1, 0], x,  z,  z,  "BEFORE — side (XZ)")
panel(gs[0, 1], xf, yf, zf, f"AFTER — top-down\n{final_mask.sum():,} pts")
panel(gs[1, 1], xf, zf, zf, "AFTER — side (XZ)")

# Artifact zone zoomed
art_bx = x[art_mask]; art_by = y[art_mask]; art_bz = z[art_mask]
art_after = final_mask & art_mask
art_ax = pts[art_after, 0]; art_ay = pts[art_after, 1]; art_az = pts[art_after, 2]

ax_b = fig.add_subplot(gs[0, 2], facecolor=BG)
ax_b.scatter(art_bx[::2], art_by[::2], c=cmap(norm_c(art_bz[::2])),
             s=0.8, alpha=0.8, linewidths=0, rasterized=True)
ax_b.set_aspect("equal")
ax_b.set_title(f"Staircase zone BEFORE\n{art_mask.sum():,} pts",
               color="white", fontsize=10, pad=5)
ax_b.tick_params(colors="#555555", labelsize=7)
for sp in ax_b.spines.values(): sp.set_edgecolor("#ff4444")

ax_a = fig.add_subplot(gs[1, 2], facecolor=BG)
if art_after.sum() > 0:
    ax_a.scatter(art_ax[::2], art_ay[::2], c=cmap(norm_c(art_az[::2])),
                 s=0.8, alpha=0.8, linewidths=0, rasterized=True)
ax_a.set_aspect("equal")
ax_a.set_title(f"Staircase zone AFTER\n{art_after.sum():,} pts",
               color="white", fontsize=10, pad=5)
ax_a.tick_params(colors="#555555", labelsize=7)
for sp in ax_a.spines.values(): sp.set_edgecolor("#44ff88")

cbar_ax = fig.add_axes([0.94, 0.1, 0.012, 0.8])
sm = ScalarMappable(cmap=cmap, norm=norm_c)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("Z (m)", color="white", fontsize=8)
cbar.ax.yaxis.set_tick_params(color="white", labelsize=7)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

plt.savefig(IMG_OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
subprocess.run(["open", IMG_OUT])
print("Done.")
