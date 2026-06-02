# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "trimesh", "pygltflib"]
# ///

"""
Step 7d - Bake a viewer-tuned transform into any floor's global transform,
          then re-export viewer GLBs.

Usage:
  uv run pipeline/07d_apply_floor_align.py --floor <floor> --theta <deg> --tx <m> --ty <m> [--tz <m>]

  --floor  which floor to adjust (basement, main_floor, top_floor, attic). Default: top_floor
  --theta  rotation around vertical (Z-up) axis, degrees
  --tx     X shift in Z-up world metres
  --ty     Y shift in Z-up world metres (Z-up Y = horizontal depth = -(Three.js Z))
  --tz     Z shift in Z-up world metres = height change (= Three.js Y nudge)

Full 3-axis form (preferred — emitted by the viewer "Get Command" button):
  uv run pipeline/07d_apply_floor_align.py --floor <floor> --matrix "m00 m01 ... m33"

  --matrix  16 floats, row-major: the raw Three.js Y-up correction matrix from
            the viewer. It is converted to Z-up world space via conjugation by
            ZUP_TO_YUP, then pre-multiplied onto global_T[floor]. Handles any
            rotation/translation (including the in-place pivot) exactly.

The correction is pre-multiplied onto global_T[floor] in Z-up space.
The viewer "Get Command" button outputs the correct flags automatically.
"""

import sys, argparse
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import numpy as np
import pickle
import subprocess
from pathlib import Path
from config import OUT

VALID_FLOORS = ["basement", "main_floor", "top_floor", "attic"]

# Three.js Y-up -> Z-up world (must match 07_export_viewer.py)
ZUP_TO_YUP = np.array([
    [1, 0,  0, 0],
    [0, 0,  1, 0],
    [0,-1,  0, 0],
    [0, 0,  0, 1],
], dtype=np.float64)

def make_correction(theta_deg, tx, ty, tz):
    theta = np.deg2rad(theta_deg)
    c, s  = np.cos(theta), np.sin(theta)
    # Rotation around Z-up vertical axis + XYZ translation
    return np.array([
        [c, -s, 0, tx],
        [s,  c, 0, ty],
        [0,  0, 1, tz],
        [0,  0, 0,  1],
    ], dtype=np.float64)

def correction_from_yup_matrix(m16):
    """Convert a raw Three.js Y-up correction matrix to its Z-up equivalent.

    The viewer node transform is  M_view = ZUP_TO_YUP @ global_T @ (...).
    Applying M_corr_yup on the left of M_view is equivalent to applying
    M_corr_zup = ZUP_TO_YUP^-1 @ M_corr_yup @ ZUP_TO_YUP on the left of global_T.
    """
    M_yup = np.array(m16, dtype=np.float64).reshape(4, 4)
    return np.linalg.inv(ZUP_TO_YUP) @ M_yup @ ZUP_TO_YUP

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", default="top_floor",
                    choices=VALID_FLOORS, help="which floor to adjust")
    ap.add_argument("--theta", type=float, default=None, help="rotation in degrees (scalar form)")
    ap.add_argument("--tx",    type=float, default=0.0,   help="X shift (metres)")
    ap.add_argument("--ty",    type=float, default=0.0,   help="Z-up Y shift (metres, horizontal depth)")
    ap.add_argument("--tz",    type=float, default=0.0,   help="Z-up Z shift (metres, height)")
    ap.add_argument("--matrix", type=str, default=None, metavar="\"m m ... m\"",
                    help="raw 4x4 Y-up correction matrix, row-major: a single "
                         "quoted string of 16 whitespace-separated floats")
    args = ap.parse_args()

    if args.matrix is None and args.theta is None:
        ap.error("provide either --matrix (16 floats) or --theta")

    if args.matrix is not None:
        matrix = [float(x) for x in args.matrix.split()]
        if len(matrix) != 16:
            ap.error(f"--matrix needs 16 floats, got {len(matrix)}")
        M_corr = correction_from_yup_matrix(matrix)
    else:
        M_corr = make_correction(args.theta, args.tx, args.ty, args.tz)

    cache = OUT / "FINAL_transforms.pkl"
    with open(cache, "rb") as f:
        data = pickle.load(f)

    old_T = data["global_T"][args.floor]
    new_T = M_corr @ old_T
    data["global_T"][args.floor] = new_T

    print(f"Applying to {args.floor}:")
    if args.matrix is not None:
        print("  matrix form (3-axis correction from viewer)")
    else:
        print(f"  theta={args.theta:.4f} deg  tx={args.tx:.4f}m  ty={args.ty:.4f}m  tz={args.tz:.4f}m")
    print(f"  Old: X={old_T[0,3]:.3f}  Y={old_T[1,3]:.3f}  Z={old_T[2,3]:.3f}")
    print(f"  New: X={new_T[0,3]:.3f}  Y={new_T[1,3]:.3f}  Z={new_T[2,3]:.3f}")

    with open(cache, "wb") as f:
        pickle.dump(data, f)
    print("Updated FINAL_transforms.pkl")

    result = subprocess.run(
        ["uv", "run", str(Path(__file__).parent / "07_export_viewer.py")],
        cwd=str(Path(__file__).parent.parent)
    )
    if result.returncode != 0:
        print("Export failed")
        sys.exit(1)
    print("Viewer GLBs updated. Reload the browser to see the change.")

if __name__ == "__main__":
    main()
