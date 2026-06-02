# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["numpy", "open3d", "trimesh", "scipy", "pygltflib", "matplotlib"]
# ///

"""
Step 8 - Staircase-guided floor registration.

Two chains, both anchored to main_floor (fixed reference):

  basement  <--  stair_main_to_basement  -->  main_floor
  top_floor <--  stair_main_to_top       -->  main_floor

For each staircase:
  1. Detect stair treads (flat horizontal surfaces at regular Z intervals).
  2. Fit a line through tread centroids -> stair heading direction.
  3. Z-fix: shift so the anchor landing aligns with main_Z.
  4. Try 2 orientations (heading +/- 180 deg ambiguity) + ICP refinement each.
  5. Take the best ICP result as T_stair.
  6. Use the stair half touching the floating floor to correct it with ICP.
  7. Export positioned staircase GLB for the viewer.
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import numpy as np
import trimesh
import open3d as o3d
import pickle
import subprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pygltflib
from pathlib import Path
from scipy.signal import find_peaks
from config import OUT, PHOTO_FLOORS, PHOTO_STAIRS

# ── Coordinate helpers ────────────────────────────────────────────────────────
GLTF_TO_ZUP = np.array([[1,0,0],[0,0,-1],[0,1,0]], dtype=np.float64)
FLIP_XY     = np.diag([-1.,-1.,1.]).astype(np.float64)
ZUP_TO_YUP  = np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]], dtype=np.float64)

def apply3(pts, R):
    return (R @ pts.T).T

def apply4(pts, T):
    h = np.hstack([pts, np.ones((len(pts), 1))])
    return (T @ h.T).T[:, :3]

def np_to_o3d(pts):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    return pc

def add_normals(pc, radius=0.12):
    pc.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))

def load_glb_pts(path, n=120_000):
    scene  = trimesh.load(str(path), force="scene")
    meshes = [g for g in scene.geometry.values()
              if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0]
    pts, _ = trimesh.sample.sample_surface(trimesh.util.concatenate(meshes), n)
    pts    = apply3(pts.astype(np.float64), GLTF_TO_ZUP)
    pts    = apply3(pts, FLIP_XY)
    return pts

# ── Stair tread detector ──────────────────────────────────────────────────────

def detect_stair_treads(pts):
    """
    Find horizontal tread surfaces in a staircase cloud.

    Returns dict with:
      heading  - unit 2D vector along the stair walking direction
      centroids - Nx3 tread centroid positions (sorted low -> high)
      n_treads  - number of treads found
    Returns None if fewer than 3 treads detected.
    """
    pc = np_to_o3d(pts)
    add_normals(pc, radius=0.12)
    normals = np.asarray(pc.normals)

    # Treads: normals pointing mostly upward
    flat_mask = np.abs(normals[:, 2]) > 0.80
    flat_pts  = pts[flat_mask]
    if len(flat_pts) < 100:
        print("  WARNING: too few flat points for tread detection")
        return None

    # Bin Z to find discrete tread levels (~2.5 cm bins)
    z      = flat_pts[:, 2]
    z_span = z.max() - z.min()
    nbins  = max(30, int(z_span / 0.025))
    counts, edges = np.histogram(z, bins=nbins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    min_sep = int(0.10 / (z_span / nbins))   # at least 10 cm between steps
    peaks, props = find_peaks(counts,
                              height=counts.max() * 0.05,
                              distance=max(1, min_sep))

    tread_centroids = []
    for pz in centres[peaks]:
        mask = np.abs(flat_pts[:, 2] - pz) < 0.04
        if mask.sum() >= 15:
            tread_centroids.append(flat_pts[mask].mean(0))

    if len(tread_centroids) < 3:
        print(f"  WARNING: only {len(tread_centroids)} treads found (need >= 3)")
        return None

    tc = np.array(tread_centroids)
    tc = tc[tc[:, 2].argsort()]   # sort bottom -> top

    # PCA on XY centroids -> primary direction = stair heading
    xy = tc[:, :2]
    _, _, Vt = np.linalg.svd(xy - xy.mean(0))
    heading = Vt[0]

    print(f"  Detected {len(tc)} treads  |  heading={np.degrees(np.arctan2(heading[1],heading[0])):.1f} deg")
    return {"heading": heading, "centroids": tc, "n_treads": len(tc)}

# ── ICP with a given initial transform ───────────────────────────────────────

def icp_from_init(src_pts, tgt_pts, T_init, max_dist=0.4, iters=200):
    src = np_to_o3d(src_pts)
    tgt = np_to_o3d(tgt_pts)
    add_normals(src, 0.12)
    add_normals(tgt, 0.12)
    result = o3d.pipelines.registration.registration_icp(
        src, tgt, max_dist, T_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iters),
    )
    return result.transformation, result.fitness, result.inlier_rmse


# ── 2-D rotation search (XY + yaw only, no Z change) ─────────────────────────

def correct_floor_2d(float_global, stair_half, float_Z, heading):
    """
    Find the best yaw + XY translation that aligns float_global with stair_half,
    without touching Z.  Uses a centroid-aligned rotation search scored by
    mean nearest-neighbour distance.
    """
    from scipy.spatial import cKDTree

    stair_xy = stair_half[:, :2]
    stair_c  = stair_xy.mean(0)

    # Use floor-level points for the float centroid (closer to stairwell Z)
    z_lo = stair_half[:, 2].min() - 0.5
    z_hi = stair_half[:, 2].max() + 0.5
    near_Z = float_global[(float_global[:, 2] >= z_lo) & (float_global[:, 2] <= z_hi)]
    float_ref = near_Z if len(near_Z) >= 200 else float_global
    float_c   = float_ref[:, :2].mean(0)

    # Candidate yaw angles: heading-based (2 candidates) + 6 uniform fallbacks
    h_angle = np.arctan2(heading[1], heading[0])
    angles  = np.unique(np.concatenate([
        [h_angle, h_angle + np.pi],
        np.linspace(0, 2*np.pi, 6, endpoint=False),
    ])) % (2 * np.pi)

    # Sub-sample for speed
    rng = np.random.default_rng(42)
    s_idx = rng.choice(len(stair_xy), min(4000, len(stair_xy)), replace=False)
    f_idx = rng.choice(len(float_ref[:, :2]), min(4000, len(float_ref)), replace=False)
    stair_sub = stair_xy[s_idx]
    float_sub = float_ref[:, :2][f_idx]
    kd = cKDTree(stair_sub)

    best_cost = np.inf
    best_T    = np.eye(4)
    for angle in angles:
        c, s = np.cos(angle), np.sin(angle)
        R    = np.array([[c, -s], [s, c]])
        t    = stair_c - R @ float_c
        rotated = (R @ float_sub.T).T + t
        cost = float(np.mean(np.minimum(kd.query(rotated)[0], 1.0)))
        if cost < best_cost:
            best_cost = cost
            T = np.eye(4)
            T[:2, :2] = R
            T[0, 3]   = t[0]
            T[1, 3]   = t[1]
            # Z stays 0 — no vertical shift
            best_T = T

    return best_T, best_cost

# ── Stair -> anchor floor registration ───────────────────────────────────────

def register_stair_to_floor(stair_pts, floor_pts, anchor_Z, tread_info):
    """
    Place the staircase relative to the anchor floor using the stair heading
    to seed orientation, then refine with ICP.

    Tests 4 candidate orientations derived from the heading (heading, heading+180,
    and the two perpendiculars as a fallback), picks best ICP fitness.
    """
    heading = tread_info["heading"]
    base_angle = np.arctan2(heading[1], heading[0])

    # Stair XY centroid (after Z-fix)
    stair_xy_c = stair_pts[:, :2].mean(0)

    # Target centroid near anchor_Z (where the stair bottom should land)
    near_anchor = floor_pts[np.abs(floor_pts[:, 2] - anchor_Z) < 0.4]
    if len(near_anchor) < 100:
        near_anchor = floor_pts
    floor_xy_c = near_anchor[:, :2].mean(0)

    candidates = [base_angle, base_angle + np.pi,
                  base_angle + np.pi/2, base_angle - np.pi/2]

    best = None
    for i, angle in enumerate(candidates):
        c, s = np.cos(angle), np.sin(angle)
        # Rotate stair about its XY centroid then translate to floor centroid
        dx = floor_xy_c[0] - (c*stair_xy_c[0] - s*stair_xy_c[1])
        dy = floor_xy_c[1] - (s*stair_xy_c[0] + c*stair_xy_c[1])
        T_init = np.array([[c,-s,0,dx],
                           [s, c,0,dy],
                           [0, 0,1, 0],
                           [0, 0,0, 1]], dtype=np.float64)

        T, fit, rmse = icp_from_init(stair_pts, floor_pts, T_init)
        print(f"    candidate {i} angle={np.degrees(angle):6.1f} deg  fitness={fit:.4f}  rmse={rmse:.4f}")

        if best is None or fit > best[0]:
            best = (fit, rmse, T)

    print(f"  Best: fitness={best[0]:.4f}  rmse={best[1]:.4f}")
    return best[2], best[0], best[1]

# ── Load pipeline state ───────────────────────────────────────────────────────
with open(OUT / "FINAL_transforms.pkl", "rb") as f:
    data = pickle.load(f)
global_T  = data["global_T"]
photo_pts = data["photo_pts"]

main_Z     = global_T["main_floor"][2, 3]
top_Z      = global_T["top_floor"][2, 3]
basement_Z = global_T["basement"][2, 3]
print(f"Floor Z: basement={basement_Z:.3f}  main={main_Z:.3f}  top={top_Z:.3f}")

main_global = apply4(photo_pts["main_floor"].astype(np.float64), global_T["main_floor"])
print(f"Main floor cloud: {len(main_global):,} pts  (fixed reference)")

# ── Per-staircase registration ────────────────────────────────────────────────
stair_configs = [
    {
        "key":         "main_to_top",
        "float_key":   "top_floor",
        "anchor_Z":    main_Z,
        "float_Z":     top_Z,
        "anchor_side": "lower",   # lower landing sits on main floor
        "float_side":  "upper",   # upper half overlaps top floor
        "viewer_name": "stair_main_to_top.glb",
    },
    {
        "key":         "main_to_basement",
        "float_key":   "basement",
        "anchor_Z":    main_Z,
        "float_Z":     basement_Z,
        "anchor_side": "upper",   # upper landing sits on main floor
        "float_side":  "lower",   # lower half overlaps basement
        "viewer_name": "stair_main_to_basement.glb",
    },
]

stair_results = {}

for cfg in stair_configs:
    name      = cfg["key"]
    float_key = cfg["float_key"]
    stair_path = PHOTO_STAIRS[name]

    print(f"\n{'='*60}")
    print(f"Staircase: {name}  ->  correcting {float_key}")

    if not stair_path.exists():
        print("  NOT FOUND -- skipping")
        continue

    # ── Sample & detect treads ───────────────────────────────────────────────
    raw = load_glb_pts(stair_path, n=120_000)
    tread_info = detect_stair_treads(raw)

    # ── Z-fix: align anchor-side landing to main_Z ───────────────────────────
    floor_span = abs(cfg["float_Z"] - cfg["anchor_Z"])
    z = raw[:, 2]
    counts, edges = np.histogram(z, bins=120)
    centres = 0.5 * (edges[:-1] + edges[1:])
    peaks, props = find_peaks(counts, height=counts.max() * 0.08,
                              distance=int(0.6 / (edges[1] - edges[0])))
    if len(peaks) >= 2:
        top2 = peaks[np.argsort(props["peak_heights"])[-2:]]
        z_lo, z_hi = sorted(centres[top2])
    else:
        z_lo, z_hi = z.min(), z.max()

    anchor_peak = z_lo if cfg["anchor_side"] == "lower" else z_hi
    z_shift     = cfg["anchor_Z"] - anchor_peak
    stair_pts   = raw.copy()
    stair_pts[:, 2] += z_shift
    print(f"  Z-fix: {z_shift:+.3f} m  (anchor landing -> {cfg['anchor_Z']:.3f})")

    # ── Register stair -> main floor ─────────────────────────────────────────
    print("  Registering stair -> main_floor ...")
    if tread_info is not None:
        T_stair, stair_fit, stair_rmse = register_stair_to_floor(
            stair_pts, main_global, cfg["anchor_Z"], tread_info)
    else:
        # Fallback: try 8 evenly-spaced orientations
        print("  No tread info -- trying 8 orientations ...")
        stair_xy_c  = stair_pts[:, :2].mean(0)
        floor_xy_c  = main_global[:, :2].mean(0)
        best = None
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            c, s = np.cos(angle), np.sin(angle)
            dx = floor_xy_c[0] - (c*stair_xy_c[0] - s*stair_xy_c[1])
            dy = floor_xy_c[1] - (s*stair_xy_c[0] + c*stair_xy_c[1])
            T_init = np.array([[c,-s,0,dx],[s,c,0,dy],[0,0,1,0],[0,0,0,1]], dtype=np.float64)
            T, fit, rmse = icp_from_init(stair_pts, main_global, T_init)
            if best is None or fit > best[0]:
                best = (fit, rmse, T)
        T_stair, stair_fit, stair_rmse = best[2], best[0], best[1]
        print(f"  Best fallback: fitness={stair_fit:.4f}  rmse={stair_rmse:.4f}")

    stair_global = apply4(stair_pts, T_stair)

    # ── Correct the floating floor ────────────────────────────────────────────
    mid_Z = (cfg["anchor_Z"] + cfg["float_Z"]) / 2.0
    if cfg["float_side"] == "upper":
        stair_half = stair_global[stair_global[:, 2] >= mid_Z]
    else:
        stair_half = stair_global[stair_global[:, 2] <= mid_Z]

    float_global = apply4(photo_pts[float_key].astype(np.float64), global_T[float_key])
    print(f"  Stair {cfg['float_side']} half: {len(stair_half):,} pts")
    print(f"  {float_key} cloud: {len(float_global):,} pts")

    if len(stair_half) >= 300 and len(float_global) >= 300:
        print(f"  Correcting {float_key} (2-D rotation search, no Z change) ...")

        h = tread_info["heading"] if tread_info else np.array([1.0, 0.0])
        M_corr, corr_cost = correct_floor_2d(float_global, stair_half, cfg["float_Z"], h)

        theta = np.degrees(np.arctan2(M_corr[1, 0], M_corr[0, 0]))
        print(f"  Correction: theta={theta:.2f} deg  tx={M_corr[0,3]:.3f}  ty={M_corr[1,3]:.3f}")
        print(f"  2D cost={corr_cost:.4f}  (Z unchanged)")

        old_T = global_T[float_key].copy()
        global_T[float_key] = M_corr @ old_T
        new_T = global_T[float_key]
        print(f"  {float_key}: X {old_T[0,3]:.3f}->{new_T[0,3]:.3f}"
              f"  Y {old_T[1,3]:.3f}->{new_T[1,3]:.3f}"
              f"  Z {old_T[2,3]:.3f}->{new_T[2,3]:.3f}")
    else:
        M_corr = np.eye(4)
        print("  Too few points -- correction skipped")

    # ── Export positioned staircase GLB ──────────────────────────────────────
    G4     = np.eye(4); G4[:3, :3]    = GLTF_TO_ZUP
    flip4  = np.eye(4); flip4[:3, :3] = FLIP_XY
    T_zfix = np.eye(4); T_zfix[2, 3]  = z_shift
    M_full = ZUP_TO_YUP @ T_stair @ T_zfix @ flip4 @ G4

    out_dir   = OUT / "viewer"; out_dir.mkdir(exist_ok=True)
    stair_out = out_dir / cfg["viewer_name"]
    gltf      = pygltflib.GLTF2().load(str(stair_path))
    wrapper   = pygltflib.Node(name=f"world_{name}",
                               matrix=M_full.T.flatten().tolist(),
                               children=list(gltf.scenes[gltf.scene].nodes))
    gltf.nodes.append(wrapper)
    gltf.scenes[gltf.scene].nodes = [len(gltf.nodes) - 1]
    gltf.save(str(stair_out))
    print(f"  GLB -> {stair_out.name}")

    stair_results[name] = {
        "stair_global": stair_global,
        "float_key":    float_key,
        "stair_fit":    stair_fit,
        "M_corr":       M_corr,
    }

# ── Save ──────────────────────────────────────────────────────────────────────
with open(OUT / "FINAL_transforms.pkl", "wb") as f:
    pickle.dump(data, f)
print("\nSaved FINAL_transforms.pkl")
print("\nFinal global_T:")
for k, T in global_T.items():
    print(f"  {k:12s}  X={T[0,3]:.3f}  Y={T[1,3]:.3f}  Z={T[2,3]:.3f}")

# ── Diagnostic ────────────────────────────────────────────────────────────────
clouds = {
    "basement":   apply4(photo_pts["basement"].astype(np.float64),   global_T["basement"]),
    "main_floor": apply4(photo_pts["main_floor"].astype(np.float64), global_T["main_floor"]),
    "top_floor":  apply4(photo_pts["top_floor"].astype(np.float64),  global_T["top_floor"]),
}
cloud_colors = {"basement": "#a8d8ea", "main_floor": "#7ec8e3", "top_floor": "#b5ead7"}
stair_colors = {"main_to_top": "#e8b86d", "main_to_basement": "#f4a261"}

S = 6
fig, axes = plt.subplots(1, 2, figsize=(22, 10), facecolor="#0d0d1a")
fig.suptitle("Step 8 - Staircase-guided registration (tread detection + ICP)",
             color="white", fontsize=12)

for ax, (xi, yi, lbl) in zip(axes, [(0, 1, "Top-down XY"), (0, 2, "Side XZ")]):
    ax.set_facecolor("#050510")
    ax.set_title(lbl, color="white", fontsize=10)
    ax.tick_params(colors="#444466", labelsize=6)
    for sp in ax.spines.values(): sp.set_edgecolor("#222244")
    for floor, pts in clouds.items():
        ax.scatter(pts[::S, xi], pts[::S, yi],
                   c=cloud_colors[floor], s=0.3, alpha=0.5, linewidths=0, label=floor)
    for name, res in stair_results.items():
        sg = res["stair_global"]
        ax.scatter(sg[::2, xi], sg[::2, yi],
                   c=stair_colors.get(name, "#ffffff"), s=1.5, alpha=0.9,
                   linewidths=0, label=f"{name} (fit={res['stair_fit']:.3f})")
    ax.set_aspect("equal")
    ax.legend(fontsize=7, facecolor="#111122", labelcolor="white",
              edgecolor="#333355", markerscale=6)

out_img = OUT / "08_stair_register.png"
plt.tight_layout()
plt.savefig(out_img, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
subprocess.run(["open", str(out_img)])
print(f"Diagnostic -> {out_img}")

# ── Re-export viewer GLBs ─────────────────────────────────────────────────────
print("\nRe-exporting viewer GLBs...")
subprocess.run(["uv", "run", str(Path(__file__).parent / "07_export_viewer.py")],
               cwd=str(Path(__file__).parent.parent))
print("Done - reload the browser.")
