# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pygltflib"]
# ///

"""
Step 7f - Apply a viewer-tuned nudge to a staircase GLB in-place.

The staircase GLBs live in output/viewer/ and are already in Three.js Y-up space.
This script patches their root node matrix directly, so no Z-up conversion is needed.

Usage (coords are all Three.js Y-up):
  uv run pipeline/07f_nudge_stair.py --stair main_to_top --theta <deg> --tx <m> --ty <m> --tz <m>

  --stair  main_to_top | main_to_basement
  --theta  rotation around Y axis (degrees)
  --tx     X shift (metres)
  --ty     Y shift / height (metres)
  --tz     Z shift / depth (metres)
"""

import sys, argparse
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import numpy as np
import pygltflib
from config import OUT

VALID = ["main_to_top", "main_to_basement"]

def correction_yup(theta_deg, tx, ty, tz):
    theta = np.deg2rad(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c,  0, -s, tx],
        [0,  1,  0, ty],
        [s,  0,  c, tz],
        [0,  0,  0,  1],
    ], dtype=np.float64)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stair",    required=True, choices=VALID)
    ap.add_argument("--theta",    type=float, required=True)
    ap.add_argument("--tx",       type=float, default=0.0)
    ap.add_argument("--ty",       type=float, default=0.0)
    ap.add_argument("--tz",       type=float, default=0.0)
    ap.add_argument("--mirror-x", action="store_true")
    ap.add_argument("--mirror-z", action="store_true")
    args = ap.parse_args()

    glb_path = OUT / "viewer" / f"stair_{args.stair}.glb"
    if not glb_path.exists():
        print(f"ERROR: {glb_path} not found — run 08_stair_register.py first")
        sys.exit(1)

    gltf = pygltflib.GLTF2().load(str(glb_path))

    # Find the world root node
    root_idx = gltf.scenes[gltf.scene].nodes[0]
    root_node = gltf.nodes[root_idx]

    if not root_node.matrix:
        print("ERROR: root node has no matrix")
        sys.exit(1)

    # GLTF matrix is column-major; reshape to row-major 4x4
    old_M = np.array(root_node.matrix, dtype=np.float64).reshape(4, 4).T

    # Apply mirror in local space (post-multiply) before world correction
    if args.mirror_x:
        old_M = old_M @ np.diag([-1., 1., 1., 1.])
    elif args.mirror_z:
        old_M = old_M @ np.diag([1., 1., -1., 1.])

    M_corr = correction_yup(args.theta, args.tx, args.ty, args.tz)
    new_M  = M_corr @ old_M

    root_node.matrix = new_M.T.flatten().tolist()
    gltf.save(str(glb_path))

    mirror_str = " mirror-x" if args.mirror_x else " mirror-z" if args.mirror_z else ""
    print(f"Nudged {glb_path.name}:")
    print(f"  theta={args.theta:.4f} deg  tx={args.tx:.4f}  ty={args.ty:.4f}  tz={args.tz:.4f}{mirror_str}")
    print(f"  Old origin: {old_M[:3, 3]}")
    print(f"  New origin: {new_M[:3, 3]}")
    print("Reload the browser to see the change.")

if __name__ == "__main__":
    main()
