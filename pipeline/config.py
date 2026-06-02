# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

"""Central path config for the pipeline — import this from any pipeline script."""

from pathlib import Path

ROOT  = Path(__file__).parent.parent
DOCS  = ROOT / "docs"
PHOTO = DOCS / "photogrammetry"
OUT   = ROOT / "output"
OUT.mkdir(exist_ok=True)

LIDAR = {
    "basement":   DOCS / "basement"   / "5_29_2026.ply",
    "main_floor": DOCS / "main_floor" / "5_29_2026.ply",
    "top_floor":  DOCS / "top_floor"  / "5_29_2026_clean.ply",
}

PHOTO_FLOORS = {
    "basement":   PHOTO / "basement"   / "Basement - 2505 NE 42nd Ave.glb",
    "main_floor": PHOTO / "main_floor" / "Main Floor - 2505 42nd Ave.glb",
    "top_floor":  PHOTO / "top_floor"  / "Second Floor - 2505 NE 42nd Ave.glb",
    "attic":      PHOTO / "attic"      / "Attic - 2505 NE 42nd Ave.glb",
}

PHOTO_STAIRS = {
    "main_to_top":      PHOTO / "stair_main_to_top"      / "Main Floor Stair - 2505 NE 42nd Ave.glb",
    "main_to_basement": PHOTO / "stair_main_to_basement" / "Main to Basement Stairway - 2505 NE 42nd Ave.glb",
}

# Floor stack order bottom → top
FLOOR_ORDER = ["basement", "main_floor", "top_floor", "attic"]
