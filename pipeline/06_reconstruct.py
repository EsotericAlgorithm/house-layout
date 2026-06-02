# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pymeshlab", "trimesh"]
# ///

"""
Step 6 — Surface reconstruction.
Estimates normals then runs Screened Poisson on the merged point cloud.
Outputs a watertight mesh as output/mesh.ply and output/mesh.glb
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import pymeshlab
import numpy as np
from config import OUT

IN_PLY  = OUT / "merged.ply"
OUT_PLY = OUT / "mesh.ply"
OUT_GLB = OUT / "mesh.glb"

print(f"Loading {IN_PLY}...")
ms = pymeshlab.MeshSet()
ms.load_new_mesh(str(IN_PLY))
print(f"  {ms.current_mesh().vertex_number():,} vertices")

print("Estimating normals...")
ms.compute_normal_for_point_clouds(k=16, smoothiter=2)

print("Running Screened Poisson reconstruction (depth=10)...")
ms.generate_surface_reconstruction_screened_poisson(
    depth=10,
    fulldepth=5,
    cgdepth=0,
    scale=1.1,
    samplespernode=1.5,
    pointweight=4.0,
    iters=8,
    confidence=False,
    preclean=True,
)
print(f"  Mesh: {ms.current_mesh().vertex_number():,} verts, "
      f"{ms.current_mesh().face_number():,} faces")

print("Cleaning mesh...")
ms.meshing_remove_unreferenced_vertices()
ms.meshing_remove_duplicate_faces()
ms.meshing_remove_connected_component_by_face_number(mincomponentsize=500)
print(f"  After clean: {ms.current_mesh().vertex_number():,} verts, "
      f"{ms.current_mesh().face_number():,} faces")

# Decimate to ~400K faces for browser performance
target_faces = 400_000
current_faces = ms.current_mesh().face_number()
if current_faces > target_faces:
    ratio = target_faces / current_faces
    print(f"Decimating {current_faces:,} → ~{target_faces:,} faces (ratio {ratio:.3f})...")
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=target_faces,
        qualitythr=0.3,
        preserveboundary=True,
        preservenormal=True,
        optimalplacement=True,
        planarquadric=True,
    )
    print(f"  After decimation: {ms.current_mesh().vertex_number():,} verts, "
          f"{ms.current_mesh().face_number():,} faces")

print(f"Saving mesh PLY → {OUT_PLY}")
ms.save_current_mesh(str(OUT_PLY))
print("Saved.")

# Convert to GLB via trimesh
import trimesh as tm
print(f"Converting to GLB → {OUT_GLB}")
mesh = tm.load(str(OUT_PLY))
mesh.export(str(OUT_GLB))
print(f"Done. Mesh: {OUT_PLY.stat().st_size//1024:,}KB PLY  |  {OUT_GLB.stat().st_size//1024:,}KB GLB")
