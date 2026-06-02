# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "trimesh", "matplotlib", "scipy"]
# ///

"""
Step 5 — Merge all registered data into a single point cloud.

For each floor:
  - Apply per-floor LiDAR→photo transform (step 3)
  - Apply global stacking transform (step 4)
  - Sample the photogrammetry mesh at higher density for texture-rich surfaces
  - Combine LiDAR + photogrammetry, voxel downsample to remove redundancy

Output: output/merged.ply — single unified point cloud, global coordinates
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import numpy as np
import trimesh
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import subprocess
from pathlib import Path
from config import OUT, PHOTO_FLOORS, FLOOR_ORDER

PHOTO_SAMPLE  = 150_000   # points sampled per floor photogrammetry mesh
VOXEL_SIZE    = 0.03      # metres — merge resolution (3cm)
GLTF_TO_ZUP  = np.array([[1,0,0],[0,0,-1],[0,1,0]], dtype=np.float64)
FLIP_XY      = np.diag([-1.,-1.,1.]).astype(np.float64)
PER_EXTRA    = {"main_floor": FLIP_XY, "top_floor": FLIP_XY}


def normalize_photo(pts, name):
    pts = (GLTF_TO_ZUP @ pts.T).T
    if name in PER_EXTRA:
        pts = (PER_EXTRA[name] @ pts.T).T
    return pts


def apply4(pts, T):
    ones = np.ones((len(pts), 1), dtype=np.float64)
    return (T @ np.hstack([pts.astype(np.float64), ones]).T).T[:, :3]


def load_glb_with_color(path, n):
    scene  = trimesh.load(str(path), force="scene")
    meshes = [g for g in scene.geometry.values()
              if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0]
    mesh   = trimesh.util.concatenate(meshes)
    pts, face_idx = trimesh.sample.sample_surface(mesh, count=n)
    if mesh.visual and hasattr(mesh.visual, "to_color"):
        colors = mesh.visual.to_color().vertex_colors[
            mesh.faces[face_idx].mean(axis=1).astype(int)][:, :3]
    else:
        colors = np.full((len(pts), 3), 180, dtype=np.uint8)
    return pts.astype(np.float64), colors


def voxel_downsample(pts, colors, voxel):
    """Grid-based downsample — keep one point per voxel cell."""
    idx  = np.floor(pts / voxel).astype(np.int32)
    keys = idx[:, 0] * 100003 + idx[:, 1] * 1009 + idx[:, 2]  # cheap hash
    _, first = np.unique(keys, return_index=True)
    return pts[first], colors[first]


cache = OUT / "FINAL_transforms.pkl"
with open(cache, "rb") as f:
    data = pickle.load(f)

global_T      = data["global_T"]
lidar_clipped = data["lidar_clipped"]
per_floor_T   = data["per_floor_T"]

all_pts    = []
all_colors = []
floors     = ["basement", "main_floor", "top_floor"]  # LiDAR floors
all_floors = floors + ["attic"]

print("Merging floors...")

for floor in all_floors:
    print(f"\n  {floor}")
    T_global = global_T[floor]

    # ── Photogrammetry ────────────────────────────────────────────────────────
    glb_path = PHOTO_FLOORS.get(floor)
    if glb_path and glb_path.exists():
        print(f"    photo: sampling {PHOTO_SAMPLE:,} pts from {glb_path.name}...")
        p_pts, p_col = load_glb_with_color(glb_path, PHOTO_SAMPLE)
        p_pts = normalize_photo(p_pts, floor)
        p_pts = apply4(p_pts, T_global)
        all_pts.append(p_pts)
        all_colors.append(p_col)
        print(f"    photo: {len(p_pts):,} pts added")

    # ── LiDAR (registered to photo space, then lifted to global) ─────────────
    if floor in lidar_clipped:
        l_pts = lidar_clipped[floor].astype(np.float64)
        T_reg = per_floor_T[floor]    # LiDAR → photo local space
        l_pts = apply4(l_pts, T_reg)
        l_pts = apply4(l_pts, T_global)
        l_col = np.full((len(l_pts), 3), 160, dtype=np.uint8)  # neutral gray
        all_pts.append(l_pts)
        all_colors.append(l_col)
        print(f"    lidar: {len(l_pts):,} pts added")

print(f"\nCombining {sum(len(p) for p in all_pts):,} total points...")
merged_pts = np.vstack(all_pts)
merged_col = np.vstack(all_colors)

print(f"Voxel downsampling at {VOXEL_SIZE*100:.0f}cm resolution...")
merged_pts, merged_col = voxel_downsample(merged_pts, merged_col, VOXEL_SIZE)
print(f"After downsample: {len(merged_pts):,} points")

# ── Write PLY ─────────────────────────────────────────────────────────────────
out_ply = OUT / "merged.ply"
print(f"\nWriting → {out_ply}")
dt = np.dtype([("x","f4"),("y","f4"),("z","f4"),
               ("red","u1"),("green","u1"),("blue","u1")])
arr = np.empty(len(merged_pts), dtype=dt)
arr["x"], arr["y"], arr["z"] = merged_pts[:,0], merged_pts[:,1], merged_pts[:,2]
arr["red"], arr["green"], arr["blue"] = merged_col[:,0], merged_col[:,1], merged_col[:,2]

with open(out_ply, "wb") as f:
    f.write(b"ply\nformat binary_little_endian 1.0\n"
            b"comment Merged house model - layout pipeline\n")
    f.write(f"element vertex {len(arr)}\n".encode())
    for name, (t, _) in [("x","f4"),("y","f4"),("z","f4")]:
        pass
    for field, tc in [("x","float"),("y","float"),("z","float"),
                      ("red","uchar"),("green","uchar"),("blue","uchar")]:
        f.write(f"property {tc} {field}\n".encode())
    f.write(b"end_header\n")
    f.write(arr.tobytes())
print("Saved.")

# ── Render preview ────────────────────────────────────────────────────────────
print("Rendering preview...")
S    = 8
norm = Normalize(vmin=merged_pts[:,2].min(), vmax=merged_pts[:,2].max())
cmap = plt.cm.plasma
BG   = "#050510"

fig, axes = plt.subplots(1, 2, figsize=(20, 9), facecolor="#0d0d1a")
fig.suptitle(f"Step 5 — Merged Point Cloud  ({len(merged_pts):,} pts)",
             color="white", fontsize=13, fontweight="bold")

for ax, (xi, yi, label) in zip(axes, [(0,1,"Top-down (XY)"),(0,2,"Side (XZ)")]):
    ax.set_facecolor(BG)
    ax.set_title(label, color="white", fontsize=10)
    ax.tick_params(colors="#444466", labelsize=6)
    for sp in ax.spines.values(): sp.set_edgecolor("#222244")
    sub = merged_pts[::S]
    ax.scatter(sub[:,xi], sub[:,yi], c=cmap(norm(sub[:,2])),
               s=0.3, alpha=0.7, linewidths=0, rasterized=True)
    ax.set_aspect("equal")

cbar_ax = fig.add_axes([0.92, 0.1, 0.012, 0.8])
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("Z (m)", color="white", fontsize=8)
cbar.ax.yaxis.set_tick_params(color="white", labelsize=7)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

out_img = OUT / "05_merged_preview.png"
plt.savefig(out_img, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
subprocess.run(["open", str(out_img)])
print(f"Preview → {out_img}")
print(f"\nDone. Merged cloud: {out_ply}")
