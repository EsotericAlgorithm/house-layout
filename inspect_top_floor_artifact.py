# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "matplotlib"]
# ///

"""
Zoomed artifact inspection for the top floor upper-right staircase region.
Shows the full cloud vs. the suspect zone, plus Z-slices through it.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle
import os
import subprocess

PLY = os.path.join(os.path.dirname(__file__), "docs", "top_floor", "5_29_2026.ply")
OUT = os.path.join(os.path.dirname(__file__), "top_floor_artifact.png")

# Upper-right quadrant — adjust if needed after seeing results
ARTIFACT_X_MIN = 1.0
ARTIFACT_Y_MIN = 1.0


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
x = data["x"].astype(np.float32)
y = data["y"].astype(np.float32)
z = data["z"].astype(np.float32)

# Separate artifact region from clean region
in_artifact = (x >= ARTIFACT_X_MIN) & (y >= ARTIFACT_Y_MIN)
out_mask = ~in_artifact
print(f"  Total points   : {len(x):,}")
print(f"  Artifact region: {in_artifact.sum():,}  (x>={ARTIFACT_X_MIN}, y>={ARTIFACT_Y_MIN})")
print(f"  Clean region   : {out_mask.sum():,}")

norm = Normalize(vmin=z.min(), vmax=z.max())
cmap = plt.cm.plasma

BG = "#050510"
fig = plt.figure(figsize=(22, 14), facecolor="#0d0d1a")
fig.suptitle("Top Floor — Staircase Artifact Detail", fontsize=15,
             color="white", fontweight="bold", y=0.98)
gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.2,
                       top=0.93, bottom=0.05, left=0.04, right=0.94)

def scatter(ax, xs, ys, zs, s=0.4, alpha=0.6):
    c = cmap(norm(zs))
    ax.scatter(xs, ys, c=c, s=s, alpha=alpha, linewidths=0, rasterized=True)

def style(ax, xlabel, ylabel, title):
    ax.set_facecolor(BG)
    ax.set_xlabel(xlabel, color="#aaaaaa", fontsize=8)
    ax.set_ylabel(ylabel, color="#aaaaaa", fontsize=8)
    ax.set_title(title, color="white", fontsize=9, pad=5)
    ax.tick_params(colors="#555555", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#222244")

# Full top-down with artifact box highlighted
ax0 = fig.add_subplot(gs[0, 0], facecolor=BG)
scatter(ax0, x[::4], y[::4], z[::4])
ax0.add_patch(Rectangle((ARTIFACT_X_MIN, ARTIFACT_Y_MIN),
                          x.max() - ARTIFACT_X_MIN, y.max() - ARTIFACT_Y_MIN,
                          linewidth=1.5, edgecolor="#ff4444", facecolor="none",
                          label="Artifact zone"))
ax0.set_aspect("equal")
style(ax0, "X (m)", "Y (m)", "Full Top-Down — red = artifact zone")

# Zoomed top-down: artifact region only
ax1 = fig.add_subplot(gs[0, 1], facecolor=BG)
mask = in_artifact
scatter(ax1, x[mask], y[mask], z[mask], s=0.5)
ax1.set_aspect("equal")
style(ax1, "X (m)", "Y (m)", "Artifact Zone — top-down (zoomed)")

# Side view XZ of artifact zone
ax2 = fig.add_subplot(gs[0, 2], facecolor=BG)
scatter(ax2, x[mask], z[mask], z[mask], s=0.5)
ax2.set_aspect("equal")
style(ax2, "X (m)", "Z (m)", "Artifact Zone — side view (XZ)")

# Side view YZ of artifact zone
ax3 = fig.add_subplot(gs[0, 3], facecolor=BG)
scatter(ax3, y[mask], z[mask], z[mask], s=0.5)
ax3.set_aspect("equal")
style(ax3, "Y (m)", "Z (m)", "Artifact Zone — front view (YZ)")

# Z histogram: full vs artifact zone
ax4 = fig.add_subplot(gs[1, 0], facecolor=BG)
ax4.hist(z, bins=200, color="#444466", alpha=0.8, label="All points", edgecolor="none")
ax4.hist(z[mask], bins=200, color="#ff6666", alpha=0.8, label="Artifact zone", edgecolor="none")
ax4.set_xlabel("Z (m)", color="#aaaaaa", fontsize=8)
ax4.set_ylabel("Count", color="#aaaaaa", fontsize=8)
ax4.set_title("Z Distribution — full vs artifact zone", color="white", fontsize=9, pad=5)
ax4.tick_params(colors="#555555", labelsize=7)
ax4.legend(fontsize=7, facecolor="#111122", labelcolor="white", edgecolor="#333355")
for spine in ax4.spines.values():
    spine.set_edgecolor("#222244")

# Z slices through artifact zone to reveal layering
z_art = z[mask]
x_art = x[mask]
y_art = y[mask]
z_min, z_max = z_art.min(), z_art.max()
n_slices = 3
slice_edges = np.linspace(z_min, z_max, n_slices + 1)

for i in range(n_slices):
    zlo, zhi = slice_edges[i], slice_edges[i + 1]
    sl = mask & (z >= zlo) & (z < zhi)
    ax = fig.add_subplot(gs[1, i + 1], facecolor=BG)
    if sl.sum() > 0:
        ax.scatter(x[sl], y[sl], c="#7ec8e3", s=0.5, alpha=0.7,
                   linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    style(ax, "X (m)", "Y (m)", f"Z slice {zlo:.2f} → {zhi:.2f} m")

# colorbar
cbar_ax = fig.add_axes([0.945, 0.35, 0.012, 0.55])
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
