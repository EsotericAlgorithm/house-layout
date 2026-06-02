# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "matplotlib"]
# ///

"""
Detailed inspection of the top floor scan to identify and locate artifacts.
Points are colored by height (Z) so structural anomalies stand out.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import os
import subprocess

PLY = os.path.join(os.path.dirname(__file__), "docs", "top_floor", "5_29_2026.ply")
OUT = os.path.join(os.path.dirname(__file__), "top_floor_inspect.png")
SUBSAMPLE = 4  # finer than mosaic for better detail


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


print("Loading top floor...")
data = parse_ply(PLY)
x = data["x"][::SUBSAMPLE].astype(np.float32)
y = data["y"][::SUBSAMPLE].astype(np.float32)
z = data["z"][::SUBSAMPLE].astype(np.float32)
print(f"  {len(x):,} points (subsampled from {len(data):,})")

# height-based colormap
norm = Normalize(vmin=z.min(), vmax=z.max())
cmap = plt.cm.plasma
colors = cmap(norm(z))

fig = plt.figure(figsize=(20, 14), facecolor="#0d0d1a")
fig.suptitle("Top Floor — Artifact Inspection (colored by height)",
             fontsize=15, color="white", fontweight="bold", y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.2,
                       top=0.93, bottom=0.05, left=0.05, right=0.93)

BG = "#050510"

def make_ax(pos, xlabel, ylabel, title, xs, ys, cs, equal=True):
    ax = fig.add_subplot(pos, facecolor=BG)
    ax.scatter(xs, ys, c=cs, s=0.5, alpha=0.7, linewidths=0, rasterized=True)
    if equal:
        ax.set_aspect("equal")
    ax.set_xlabel(xlabel, color="#aaaaaa", fontsize=8)
    ax.set_ylabel(ylabel, color="#aaaaaa", fontsize=8)
    ax.set_title(title, color="white", fontsize=10, pad=5)
    ax.tick_params(colors="#555555", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#222244")
    return ax

# Top-down XY (height = color)
make_ax(gs[0, 0], "X (m)", "Y (m)", "Top-Down (XY) — color = height", x, y, colors)

# Side XZ
make_ax(gs[0, 1], "X (m)", "Z/height (m)", "Side View (XZ)", x, z, colors)

# Front YZ
make_ax(gs[0, 2], "Y (m)", "Z/height (m)", "Front View (YZ)", y, z, colors)

# Z histogram — spikes reveal duplicate layers
ax_hist = fig.add_subplot(gs[1, 0], facecolor=BG)
ax_hist.hist(z, bins=200, color="#7ec8e3", alpha=0.8, edgecolor="none")
ax_hist.set_xlabel("Z / height (m)", color="#aaaaaa", fontsize=8)
ax_hist.set_ylabel("Point count", color="#aaaaaa", fontsize=8)
ax_hist.set_title("Z Distribution — double peaks = duplicate layers", color="white", fontsize=10, pad=5)
ax_hist.tick_params(colors="#555555", labelsize=7)
for spine in ax_hist.spines.values():
    spine.set_edgecolor("#222244")
ax_hist.set_facecolor(BG)

# X histogram
ax_xhist = fig.add_subplot(gs[1, 1], facecolor=BG)
ax_xhist.hist(x, bins=200, color="#e8b86d", alpha=0.8, edgecolor="none")
ax_xhist.set_xlabel("X (m)", color="#aaaaaa", fontsize=8)
ax_xhist.set_ylabel("Point count", color="#aaaaaa", fontsize=8)
ax_xhist.set_title("X Distribution — gaps = room boundaries", color="white", fontsize=10, pad=5)
ax_xhist.tick_params(colors="#555555", labelsize=7)
for spine in ax_xhist.spines.values():
    spine.set_edgecolor("#222244")
ax_xhist.set_facecolor(BG)

# Y histogram
ax_yhist = fig.add_subplot(gs[1, 2], facecolor=BG)
ax_yhist.hist(y, bins=200, color="#b5ead7", alpha=0.8, edgecolor="none")
ax_yhist.set_xlabel("Y (m)", color="#aaaaaa", fontsize=8)
ax_yhist.set_ylabel("Point count", color="#aaaaaa", fontsize=8)
ax_yhist.set_title("Y Distribution", color="white", fontsize=10, pad=5)
ax_yhist.tick_params(colors="#555555", labelsize=7)
for spine in ax_yhist.spines.values():
    spine.set_edgecolor("#222244")
ax_yhist.set_facecolor(BG)

# colorbar
cbar_ax = fig.add_axes([0.94, 0.35, 0.012, 0.55])
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("Height Z (m)", color="white", fontsize=8)
cbar.ax.yaxis.set_tick_params(color="white", labelsize=7)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

print(f"Saving → {OUT}")
plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print("Done.")
subprocess.run(["open", OUT])
