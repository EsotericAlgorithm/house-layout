# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "trimesh", "matplotlib", "scipy"]
# ///

"""
Step 1 — Load all photogrammetry GLBs and LiDAR PLYs, sample point clouds,
and render a combined inspection mosaic so we can assess mesh quality and
rough spatial relationships before registration.
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import subprocess
from pathlib import Path
from config import LIDAR, PHOTO_FLOORS, PHOTO_STAIRS, FLOOR_ORDER, OUT

PHOTO_SAMPLE_COUNT = 80_000
SUBSAMPLE          = 4
BG                 = "#050510"
CMAP               = plt.cm.plasma


def load_glb_as_pointcloud(path: Path, n: int) -> np.ndarray:
    scene = trimesh.load(str(path), force="scene")
    meshes = [g for g in scene.geometry.values()
              if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0]
    if not meshes:
        raise ValueError(f"No usable meshes in {path.name}")
    mesh = trimesh.util.concatenate(meshes)
    pts, _ = trimesh.sample.sample_surface(mesh, count=n)
    return pts.astype(np.float32)


def load_ply(path: Path) -> np.ndarray:
    props, n_verts = [], 0
    with open(path, "rb") as f:
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            if line == "end_header":
                break
            if line.startswith("element vertex"):
                n_verts = int(line.split()[-1])
            if line.startswith("property"):
                p = line.split(); props.append((p[1], p[2]))
        tm = {"double": "f8", "float": "f4", "uchar": "u1", "uint8": "u1",
              "int": "i4", "uint": "u4"}
        dt = np.dtype([(n, tm[t]) for t, n in props])
        data = np.frombuffer(f.read(n_verts * dt.itemsize), dtype=dt)
    return np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)


def scatter_panel(ax, pts, title, color=None, s=0.3, view="xy"):
    xs = pts[::SUBSAMPLE, 0]
    ys = pts[::SUBSAMPLE, 1] if view == "xy" else pts[::SUBSAMPLE, 2]
    zs = pts[::SUBSAMPLE, 2]
    c  = color if color is not None else CMAP(Normalize()(zs))
    ax.scatter(xs, ys, c=c, s=s, alpha=0.6, linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    ax.set_facecolor(BG)
    ax.set_title(title, color="white", fontsize=8, pad=4)
    ax.tick_params(colors="#444466", labelsize=6)
    for sp in ax.spines.values(): sp.set_edgecolor("#222244")


print("Loading photogrammetry GLBs...")
photo_pts = {}
for name, path in {**PHOTO_FLOORS, **PHOTO_STAIRS}.items():
    print(f"  {name}...", end=" ", flush=True)
    pts = load_glb_as_pointcloud(path, PHOTO_SAMPLE_COUNT)
    photo_pts[name] = pts
    bb = pts.max(0) - pts.min(0)
    print(f"{len(pts):,} pts  bbox {bb[0]:.1f}×{bb[1]:.1f}×{bb[2]:.1f}m")

print("\nLoading LiDAR PLYs...")
lidar_pts = {}
for name, path in LIDAR.items():
    print(f"  {name}...", end=" ", flush=True)
    pts = load_ply(path)
    lidar_pts[name] = pts
    bb = pts.max(0) - pts.min(0)
    print(f"{len(pts):,} pts  bbox {bb[0]:.1f}×{bb[1]:.1f}×{bb[2]:.1f}m")

# ── Render ────────────────────────────────────────────────────────────────────
FLOOR_COL  = {"basement": "#e8b86d", "main_floor": "#7ec8e3",
               "top_floor": "#b5ead7", "attic": "#c9b1ff"}
STAIR_COL  = {"main_to_top": "#ff9980", "main_to_basement": "#ffcc80"}

rows = len(FLOOR_ORDER) + 1  # floors + staircase row
fig = plt.figure(figsize=(16, rows * 4 + 1), facecolor="#0d0d1a")
fig.suptitle("Pipeline Step 1 — Photogrammetry + LiDAR Inspection",
             color="white", fontsize=13, fontweight="bold", y=0.99)
gs  = gridspec.GridSpec(rows, 3, figure=fig, hspace=0.45, wspace=0.2,
                        top=0.96, bottom=0.02, left=0.04, right=0.96)

for row, floor in enumerate(FLOOR_ORDER):
    col = FLOOR_COL[floor]
    # photogrammetry top-down
    if floor in photo_pts:
        ax = fig.add_subplot(gs[row, 0])
        scatter_panel(ax, photo_pts[floor], f"{floor} — photo top-down", color=col)
    # photogrammetry side
    if floor in photo_pts:
        ax = fig.add_subplot(gs[row, 1])
        scatter_panel(ax, photo_pts[floor], f"{floor} — photo side (XZ)",
                      color=col, view="xz")
    # LiDAR top-down
    if floor in lidar_pts:
        ax = fig.add_subplot(gs[row, 2])
        scatter_panel(ax, lidar_pts[floor], f"{floor} — LiDAR top-down",
                      color=col, s=0.2)

# Staircase row
stair_row = len(FLOOR_ORDER)
for col_i, (name, pts) in enumerate(
        {k: photo_pts[k] for k in PHOTO_STAIRS if k in photo_pts}.items()):
    ax = fig.add_subplot(gs[stair_row, col_i])
    scatter_panel(ax, pts, f"stair: {name}", color=STAIR_COL.get(name, "#ffffff"))
    ax2 = fig.add_subplot(gs[stair_row, col_i + 1])
    scatter_panel(ax2, pts, f"stair: {name} — side", color=STAIR_COL.get(name, "#ffffff"), view="xz")
    break  # two staircases share the row

out_img = OUT / "01_inspect.png"
plt.savefig(out_img, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nSaved → {out_img}")
subprocess.run(["open", str(out_img)])

# ── Save sampled point clouds for next steps ──────────────────────────────────
import pickle
cache = OUT / "01_pointclouds.pkl"
with open(cache, "wb") as f:
    pickle.dump({"photo": photo_pts, "lidar": lidar_pts}, f)
print(f"Point cloud cache → {cache}")
