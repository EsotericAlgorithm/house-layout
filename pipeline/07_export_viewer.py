# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "trimesh", "pygltflib"]
# ///

"""
Step 7 — Export pre-positioned GLBs for the Three.js viewer.

Patches each floor's photogrammetry GLB with a root node transform that
bakes the full pipeline chain:
    GLTF Y-up → Z-up → optional flip → global_T → Y-up (Three.js)

This modifies only the GLTF JSON chunk (adds one wrapper node). The binary
chunk — geometry, UVs, and texture atlases — is preserved byte-for-byte.

Output: output/viewer/{floor}.glb  (ready to load in Three.js)
        output/viewer/meta.json    (floor Y heights for camera positioning)
        output/viewer/merged.ply   (copy of point cloud, Z-up)
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import json
import shutil
import numpy as np
import pickle
import pygltflib
from config import OUT, PHOTO_FLOORS

with open(OUT / "FINAL_transforms.pkl", "rb") as f:
    data = pickle.load(f)
global_T = data["global_T"]

# ── Coordinate transforms (must match 05_merge.py) ───────────────────────────
# Polycam GLBs are GLTF Y-up; this rotates them to Z-up (same as pipeline)
GLTF_TO_ZUP = np.array([
    [1, 0, 0, 0],
    [0, 0,-1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
], dtype=np.float64)

FLIP_XY = np.diag([-1., -1., 1., 1.]).astype(np.float64)
PER_EXTRA = {"main_floor": FLIP_XY, "top_floor": FLIP_XY}

# Rotate Z-up → Three.js Y-up: +90° around X  (Z→Y, Y→-Z)
ZUP_TO_YUP = np.array([
    [1, 0,  0, 0],
    [0, 0,  1, 0],
    [0,-1,  0, 0],
    [0, 0,  0, 1],
], dtype=np.float64)

out_dir = OUT / "viewer"
out_dir.mkdir(exist_ok=True)

meta = {}

for floor, glb_path in PHOTO_FLOORS.items():
    if not glb_path or not glb_path.exists():
        print(f"  {floor}: NOT FOUND — {glb_path}")
        continue

    flip = PER_EXTRA.get(floor, np.eye(4))
    M = ZUP_TO_YUP @ global_T[floor] @ flip @ GLTF_TO_ZUP

    # Floor origin (GLTF local 0,0,0) maps to this Y-up world position:
    origin_world = M @ np.array([0, 0, 0, 1])
    meta[floor] = {
        "y_floor": round(float(origin_world[1]), 3),
        "x":       round(float(origin_world[0]), 3),
        "z":       round(float(origin_world[2]), 3),
    }

    print(f"\n{floor}")
    print(f"  source  : {glb_path.name}  ({glb_path.stat().st_size//1024:,} KB)")
    print(f"  origin Y: {origin_world[1]:.3f} m  (eye-level target: {origin_world[1]+1.65:.2f} m)")

    # Load GLB — pygltflib reads JSON + binary chunks without decompressing geometry
    gltf = pygltflib.GLTF2().load(str(glb_path))

    # Add a wrapper root node carrying our world-space transform.
    # GLTF matrix is column-major: M.T flattened in C order = columns of M in order.
    wrapper = pygltflib.Node(
        name=f"world_{floor}",
        matrix=M.T.flatten().tolist(),
        children=list(gltf.scenes[gltf.scene].nodes),
    )
    gltf.nodes.append(wrapper)
    gltf.scenes[gltf.scene].nodes = [len(gltf.nodes) - 1]

    out_path = out_dir / f"{floor}.glb"
    gltf.save(str(out_path))
    print(f"  → {out_path.name}  ({out_path.stat().st_size//1024:,} KB)")

# Copy point cloud for viewer to reference (kept in Z-up; viewer rotates it)
ply_src = OUT / "merged.ply"
if ply_src.exists():
    shutil.copy(ply_src, out_dir / "merged.ply")
    print(f"\nCopied merged.ply  ({ply_src.stat().st_size//1024:,} KB)")

meta_path = out_dir / "meta.json"
meta_path.write_text(json.dumps(meta, indent=2))
print(f"meta.json → {meta_path}")
print("\nAll viewer assets ready in:", out_dir)
print("\nFloor origins in Three.js Y-up space:")
for floor, m in meta.items():
    print(f"  {floor:12s}: Y={m['y_floor']:.3f}  X={m['x']:.3f}  Z={m['z']:.3f}")
