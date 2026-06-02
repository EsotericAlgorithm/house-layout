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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", default="top_floor",
                    choices=VALID_FLOORS, help="which floor to adjust")
    ap.add_argument("--theta", type=float, required=True, help="rotation in degrees")
    ap.add_argument("--tx",    type=float, default=0.0,   help="X shift (metres)")
    ap.add_argument("--ty",    type=float, default=0.0,   help="Z-up Y shift (metres, horizontal depth)")
    ap.add_argument("--tz",    type=float, default=0.0,   help="Z-up Z shift (metres, height)")
    args = ap.parse_args()

    M_corr = make_correction(args.theta, args.tx, args.ty, args.tz)

    cache = OUT / "FINAL_transforms.pkl"
    with open(cache, "rb") as f:
        data = pickle.load(f)

    old_T = data["global_T"][args.floor]
    new_T = M_corr @ old_T
    data["global_T"][args.floor] = new_T

    print(f"Applying to {args.floor}:")
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
