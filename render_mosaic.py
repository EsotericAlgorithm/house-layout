# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy",
#   "matplotlib",
# ]
# ///

"""
Renders a mosaic of all three scans (top-down and side views) for floor identification.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import subprocess

SCAN_DIR = os.path.join(os.path.dirname(__file__), "docs")
SCANS = [
    ("main_floor", os.path.join(SCAN_DIR, "main_floor", "5_29_2026.ply"), "Main Floor"),
    ("basement",   os.path.join(SCAN_DIR, "basement",   "5_29_2026.ply"), "Basement"),
    ("top_floor",  os.path.join(SCAN_DIR, "top_floor",  "5_29_2026.ply"), "Top Floor"),
]
OUT = os.path.join(os.path.dirname(__file__), "scan_mosaic.png")
SUBSAMPLE = 8


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
    return data


print("Loading and rendering scans...")

fig = plt.figure(figsize=(18, 12), facecolor="#1a1a2e")
fig.suptitle("Point Cloud Scan Assessment — Confirm Floor Labels",
             fontsize=16, color="white", fontweight="bold", y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.15,
                       top=0.93, bottom=0.04, left=0.04, right=0.96)

FLOOR_COLORS = ["#e8b86d", "#7ec8e3", "#b5ead7"]

for col, (name, path, label) in enumerate(SCANS):
    print(f"  {name}...")
    data = parse_ply(path)
    x = data["x"][::SUBSAMPLE].astype(np.float32)
    y = data["y"][::SUBSAMPLE].astype(np.float32)
    z = data["z"][::SUBSAMPLE].astype(np.float32)

    has_color = ("red" in data.dtype.names and
                 (data["red"].max() > 0 or data["green"].max() > 0))
    if has_color:
        r = np.clip(data["red"][::SUBSAMPLE] / 255.0 * 1.4, 0, 1).astype(np.float32)
        g = np.clip(data["green"][::SUBSAMPLE] / 255.0 * 1.4, 0, 1).astype(np.float32)
        b = np.clip(data["blue"][::SUBSAMPLE] / 255.0 * 1.4, 0, 1).astype(np.float32)
        rgb = np.stack([r, g, b], axis=1)
    else:
        rgb = FLOOR_COLORS[col]

    color = FLOOR_COLORS[col]

    ax_top = fig.add_subplot(gs[0, col], facecolor="#0d0d1a")
    ax_top.scatter(x, y, c=rgb if has_color else color,
                   s=0.3, alpha=0.6, linewidths=0, rasterized=True)
    ax_top.set_aspect("equal")
    ax_top.set_title(f"{name}  ·  {len(data):,} pts\n{label}",
                     color="white", fontsize=11, pad=6)
    ax_top.set_xlabel("X (m)", color="#aaaaaa", fontsize=8)
    ax_top.set_ylabel("Y (m)", color="#aaaaaa", fontsize=8)
    ax_top.tick_params(colors="#666666", labelsize=7)
    for spine in ax_top.spines.values():
        spine.set_edgecolor("#333355")
    ax_top.text(0.02, 0.97, "TOP-DOWN", transform=ax_top.transAxes,
                color=color, fontsize=7, va="top", fontweight="bold")

    ax_side = fig.add_subplot(gs[1, col], facecolor="#0d0d1a")
    ax_side.scatter(x, z, c=rgb if has_color else color,
                    s=0.3, alpha=0.6, linewidths=0, rasterized=True)
    ax_side.set_aspect("equal")
    ax_side.set_xlabel("X (m)", color="#aaaaaa", fontsize=8)
    ax_side.set_ylabel("Z / height (m)", color="#aaaaaa", fontsize=8)
    ax_side.tick_params(colors="#666666", labelsize=7)
    for spine in ax_side.spines.values():
        spine.set_edgecolor("#333355")
    ax_side.text(0.02, 0.97, "SIDE VIEW", transform=ax_side.transAxes,
                color=color, fontsize=7, va="top", fontweight="bold")
    span_x = x.max() - x.min()
    span_y = y.max() - y.min()
    span_z = z.max() - z.min()
    ax_side.text(0.98, 0.03,
                 f"{span_x:.1f}m × {span_y:.1f}m × {span_z:.1f}m tall",
                 transform=ax_side.transAxes, color="#888888",
                 fontsize=7, ha="right", va="bottom")

print(f"Saving → {OUT}")
plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print("Done.")
subprocess.run(["open", OUT])
