# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "trimesh", "matplotlib", "scipy"]
# ///

"""
Shows a grid of candidate transforms for main_floor and top_floor photogrammetry
so we can visually identify which one aligns with the LiDAR orientation.
Each option is labeled — just tell me the number that matches.
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import subprocess
from config import OUT

# Base GLTF→Z-up (already applied in step 2)
# These are additional transforms applied on top
CANDIDATES = {
    1:  np.diag([ 1,  1,  1]).astype(np.float32),   # current (no change)
    2:  np.diag([-1,  1,  1]).astype(np.float32),   # flip X
    3:  np.diag([ 1, -1,  1]).astype(np.float32),   # flip Y
    4:  np.diag([ 1,  1, -1]).astype(np.float32),   # flip Z
    5:  np.diag([-1, -1,  1]).astype(np.float32),   # flip X+Y (rot 180° Z)
    6:  np.diag([-1,  1, -1]).astype(np.float32),   # flip X+Z
    7:  np.diag([ 1, -1, -1]).astype(np.float32),   # flip Y+Z
    8:  np.array([[0,1,0],[1,0,0],[0,0,1]], np.float32),   # swap X↔Y
    9:  np.array([[0,-1,0],[1,0,0],[0,0,1]], np.float32),  # swap X↔Y + flip X
    10: np.array([[0,1,0],[-1,0,0],[0,0,1]], np.float32),  # swap X↔Y + flip Y
    11: np.array([[0,-1,0],[-1,0,0],[0,0,1]], np.float32), # swap X↔Y + flip both
    12: np.array([[1,0,0],[0,0,1],[0,1,0]], np.float32),   # swap Y↔Z
}

LABELS = {
    1:  "current",
    2:  "flip X",
    3:  "flip Y",
    4:  "flip Z",
    5:  "flip X+Y",
    6:  "flip X+Z",
    7:  "flip Y+Z",
    8:  "swap X↔Y",
    9:  "swap X↔Y, flip X",
    10: "swap X↔Y, flip Y",
    11: "swap X↔Y, flip both",
    12: "swap Y↔Z",
}

cache = OUT / "02_pointclouds_normalized.pkl"
with open(cache, "rb") as f:
    data = pickle.load(f)

photo = data["photo"]
lidar = data["lidar"]

BG = "#050510"
S  = 5

for floor in ["main_floor", "top_floor"]:
    pts_photo = photo[floor]
    pts_lidar = lidar[floor]

    ncols = 4
    nrows = (len(CANDIDATES) // ncols) + 2  # +2 for LiDAR reference row + overflow
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4, nrows * 3.5),
                             facecolor="#0d0d1a")
    fig.suptitle(f"{floor} — Pick the photo transform that matches LiDAR shape\n"
                 f"(LiDAR shown top-left for reference)",
                 color="white", fontsize=12, fontweight="bold")

    axes_flat = axes.flatten()
    for ax in axes_flat:
        ax.set_facecolor(BG)
        ax.tick_params(colors="#444466", labelsize=6)
        for sp in ax.spines.values(): sp.set_edgecolor("#222244")

    # LiDAR reference (top-left)
    ax = axes_flat[0]
    ax.scatter(pts_lidar[::S, 0], pts_lidar[::S, 1],
               c="#e8b86d", s=0.4, alpha=0.7, linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    ax.set_title("LiDAR reference (top-down)", color="#e8b86d", fontsize=9, fontweight="bold")

    # Candidate transforms
    for i, (num, R) in enumerate(CANDIDATES.items()):
        ax = axes_flat[i + 1]
        transformed = (R @ pts_photo.T).T
        ax.scatter(transformed[::S, 0], transformed[::S, 1],
                   c="#7ec8e3", s=0.3, alpha=0.7, linewidths=0, rasterized=True)
        ax.set_aspect("equal")
        ax.set_title(f"#{num}: {LABELS[num]}", color="white", fontsize=8)

    # Hide unused axes
    for ax in axes_flat[len(CANDIDATES) + 1:]:
        ax.set_visible(False)

    plt.tight_layout()
    out_img = OUT / f"02b_transform_picker_{floor}.png"
    plt.savefig(out_img, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {out_img}")
    subprocess.run(["open", str(out_img)])
