# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "open3d", "trimesh", "scipy", "pygltflib", "matplotlib"]
# ///

"""
CloudCompare IPC client + registration driver.

Commands
--------
  uv run pipeline/cc_client.py repl          interactive Python→CC REPL
  uv run pipeline/cc_client.py export-ply    sample floors to global-space PLYs
  uv run pipeline/cc_client.py load          tell CC to load the exported PLYs
  uv run pipeline/cc_client.py register      FPFH+RANSAC+ICP staircase→main floor
  uv run pipeline/cc_client.py apply         save best transform → FINAL_transforms.pkl
  uv run pipeline/cc_client.py status        print current global_T values

Requires cc_server.py running inside CloudCompare's Python Console.
(export-ply, register, apply, status work without CC.)
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import json, socket, pickle, subprocess
import numpy as np
from pathlib import Path
from config import OUT, PHOTO_FLOORS, PHOTO_STAIRS

SOCK_PATH = "/tmp/cc_ipc.sock"
SEP       = b"\n##END##\n"

PLY_MAIN  = OUT / "cc_main_floor.ply"
PLY_TOP   = OUT / "cc_top_floor.ply"
PLY_STAIR = OUT / "cc_staircase.ply"
BEST_T    = OUT / "cc_best_transform.npy"   # saved between runs


# ── IPC helpers ───────────────────────────────────────────────────────────────

def _cc(code: str) -> dict:
    """Send Python code to CloudCompare; return {'ok': bool, 'result': str}."""
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(SOCK_PATH)
        sock.sendall(json.dumps({"code": code}).encode() + SEP)
        buf = b""
        while SEP not in buf:
            chunk = sock.recv(8192)
            if not chunk:
                break
            buf += chunk
        sock.close()
        return json.loads(buf.split(SEP)[0])
    except ConnectionRefusedError:
        return {"ok": False, "result": "CC not connected — run cc_server.py in CloudCompare first"}
    except Exception as exc:
        return {"ok": False, "result": str(exc)}


def cc(code: str) -> str:
    """Execute code in CC, print result, return result string."""
    r = _cc(code)
    tag = "✓" if r["ok"] else "✗"
    print(f"CC {tag}  {r['result']}")
    return r["result"]


# ── PLY export ────────────────────────────────────────────────────────────────

GLTF_TO_ZUP = np.array([[1,0,0],[0,0,-1],[0,1,0]], dtype=np.float64)
FLIP_XY     = np.diag([-1.,-1.,1.]).astype(np.float64)


def _apply3(pts, R):
    return (R @ pts.T).T


def _apply4(pts, T):
    h = np.hstack([pts, np.ones((len(pts), 1))])
    return (T @ h.T).T[:, :3]


def _sample_glb(path, n=120_000, flip=FLIP_XY):
    import trimesh
    scene  = trimesh.load(str(path), force="scene")
    meshes = [g for g in scene.geometry.values()
              if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0]
    merged = trimesh.util.concatenate(meshes)
    pts, _ = trimesh.sample.sample_surface(merged, n)
    pts    = pts.astype(np.float64)
    pts    = _apply3(pts, GLTF_TO_ZUP)
    if flip is not None:
        pts = _apply3(pts, flip)
    return pts


def _write_ply(path: Path, pts: np.ndarray):
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\n"
                f"property float x\nproperty float y\nproperty float z\n"
                f"end_header\n")
        np.savetxt(f, pts, fmt="%.6f")
    print(f"  wrote {len(pts):,} pts → {path.name}")


def cmd_export_ply():
    """Sample each floor GLB → global Z-up space → PLY for CloudCompare."""
    with open(OUT / "FINAL_transforms.pkl", "rb") as f:
        data = pickle.load(f)
    gT = data["global_T"]

    print("Sampling floors (this takes ~30 s per floor)…")
    for floor, out_path in [("main_floor", PLY_MAIN), ("top_floor", PLY_TOP)]:
        glb = PHOTO_FLOORS[floor]
        if not glb.exists():
            print(f"  {floor}: NOT FOUND — {glb}")
            continue
        pts = _sample_glb(glb, n=150_000)
        pts = _apply4(pts, gT[floor])
        _write_ply(out_path, pts)

    stair_glb = PHOTO_STAIRS["main_to_top"]
    if stair_glb.exists():
        print("Sampling staircase…")
        stair_raw = _sample_glb(stair_glb, n=80_000)
        _write_ply(PLY_STAIR, stair_raw)
    else:
        print(f"  staircase NOT FOUND — {stair_glb}")

    print("\nDone. Run:  uv run pipeline/cc_client.py load")


# ── CC load command ───────────────────────────────────────────────────────────

def cmd_load():
    for label, path in [("main_floor", PLY_MAIN), ("top_floor", PLY_TOP), ("staircase", PLY_STAIR)]:
        if not path.exists():
            print(f"  {label}: missing — run export-ply first")
            continue
        cc(f"""
entities = CC.loadFile(r"{path}")
if entities:
    entities[0].setName("{label}")
    CC.redrawAll()
_result = "loaded {label}"
""")


# ── FPFH + RANSAC + ICP registration ─────────────────────────────────────────

def cmd_register():
    """
    1. Load exported PLYs with Open3D.
    2. FPFH feature extraction + RANSAC global registration (stair → main floor).
    3. Point-to-plane ICP refinement.
    4. Derive top-floor correction from staircase upper half vs top_floor cloud.
    5. Save best transform to cc_best_transform.npy for 'apply'.
    6. Visualise in CloudCompare (optional).
    """
    import open3d as o3d
    from scipy.signal import find_peaks

    if not PLY_MAIN.exists() or not PLY_STAIR.exists():
        print("Run 'export-ply' first.")
        return

    with open(OUT / "FINAL_transforms.pkl", "rb") as f:
        data = pickle.load(f)
    gT     = data["global_T"]
    main_Z = gT["main_floor"][2, 3]
    top_Z  = gT["top_floor"][2, 3]
    print(f"Floor Z: main={main_Z:.3f}  top={top_Z:.3f}")

    # ── Load clouds ──────────────────────────────────────────────────────────
    print("\nLoading PLYs…")
    tgt_o3d = o3d.io.read_point_cloud(str(PLY_MAIN))
    src_raw  = o3d.io.read_point_cloud(str(PLY_STAIR))
    top_o3d  = o3d.io.read_point_cloud(str(PLY_TOP))

    src_pts = np.asarray(src_raw.points)

    # ── Z-fix: align staircase lower landing to main_Z ───────────────────────
    z = src_pts[:, 2]
    counts, edges = np.histogram(z, bins=120)
    centres = 0.5 * (edges[:-1] + edges[1:])
    peaks, props = find_peaks(counts, height=counts.max() * 0.08,
                              distance=int(0.6 / (edges[1] - edges[0])))
    if len(peaks) >= 2:
        top2 = peaks[np.argsort(props["peak_heights"])[-2:]]
        z_lo = sorted(centres[top2])[0]
        z_shift = main_Z - z_lo
    else:
        z_shift = 0.0
    print(f"Z-fix shift: {z_shift:+.3f} m")
    src_pts[:, 2] += z_shift
    src_o3d = o3d.geometry.PointCloud()
    src_o3d.points = o3d.utility.Vector3dVector(src_pts)

    # ── Downsample + normals + FPFH ──────────────────────────────────────────
    vox = 0.04
    src_d = src_o3d.voxel_down_sample(vox)
    tgt_d = tgt_o3d.voxel_down_sample(vox)

    for pc in (src_d, tgt_d):
        pc.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=vox * 2, max_nn=30))

    def fpfh(pc):
        return o3d.pipelines.registration.compute_fpfh_feature(
            pc, o3d.geometry.KDTreeSearchParamHybrid(radius=vox * 5, max_nn=100))

    print("Computing FPFH features…")
    src_feat = fpfh(src_d)
    tgt_feat = fpfh(tgt_d)

    # ── RANSAC global registration ────────────────────────────────────────────
    print("RANSAC global registration…")
    dist = vox * 1.5
    result_r = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_d, tgt_d, src_feat, tgt_feat,
        mutual_filter=True,
        max_correspondence_distance=dist,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(400_000, 0.9999),
    )
    T_rough = result_r.transformation
    print(f"RANSAC fitness={result_r.fitness:.4f}  rmse={result_r.inlier_rmse:.4f}")

    # ── ICP refinement ────────────────────────────────────────────────────────
    print("ICP refinement (point-to-plane)…")
    result_icp = o3d.pipelines.registration.registration_icp(
        src_o3d, tgt_o3d, vox * 0.5, T_rough,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200),
    )
    T_stair = result_icp.transformation
    print(f"ICP fitness={result_icp.fitness:.4f}  rmse={result_icp.inlier_rmse:.4f}")
    print("Stair transform (Z-up):")
    print(np.round(T_stair, 4))

    # ── Position staircase globally ───────────────────────────────────────────
    stair_global = _apply4(src_pts, T_stair)

    # ── Derive top-floor correction from stair upper half ────────────────────
    mid_Z      = (main_Z + top_Z) / 2
    stair_top  = stair_global[stair_global[:, 2] >= mid_Z]
    top_pts    = np.asarray(top_o3d.points)       # already in global_T["top_floor"] space
    z_lo_s, z_hi_s = stair_top[:, 2].min(), stair_top[:, 2].max()
    top_near   = top_pts[(top_pts[:, 2] >= z_lo_s - 0.3) & (top_pts[:, 2] <= z_hi_s + 0.5)]

    print(f"\nStair upper half: {len(stair_top):,} pts")
    print(f"Top floor near stairwell: {len(top_near):,} pts")

    if len(stair_top) >= 500 and len(top_near) >= 500:
        # FPFH+RANSAC between stair_top and top_near (XY plane, Z constrained)
        def _o3d_from_np(pts):
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(pts)
            return pc

        src_top = _o3d_from_np(stair_top)
        tgt_top = _o3d_from_np(top_near)
        src_top_d = src_top.voxel_down_sample(vox)
        tgt_top_d = tgt_top.voxel_down_sample(vox)
        for pc in (src_top_d, tgt_top_d):
            pc.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=vox * 2, max_nn=30))
        src_top_f = fpfh(src_top_d)
        tgt_top_f = fpfh(tgt_top_d)

        print("RANSAC for top-floor correction…")
        r2 = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            tgt_top_d, src_top_d, tgt_top_f, src_top_f,
            mutual_filter=True,
            max_correspondence_distance=dist,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=4,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist),
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(400_000, 0.9999),
        )
        r2_icp = o3d.pipelines.registration.registration_icp(
            tgt_top, src_top, vox * 0.5, r2.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        )
        M_corr = r2_icp.transformation   # top_floor → stair-guided target
        print(f"Top-floor correction: fitness={r2_icp.fitness:.4f}  rmse={r2_icp.inlier_rmse:.4f}")
    else:
        M_corr = np.eye(4)
        print("Too few points — top-floor correction skipped (identity)")

    # ── Save best transform ───────────────────────────────────────────────────
    result = {
        "T_stair":  T_stair,
        "M_corr":   M_corr,
        "z_shift":  z_shift,
        "fitness":  float(result_icp.fitness),
        "rmse":     float(result_icp.inlier_rmse),
    }
    np.save(str(BEST_T), result, allow_pickle=True)
    print(f"\nSaved transform → {BEST_T.name}")

    # Report the correction in degrees/metres
    theta_corr = np.degrees(np.arctan2(M_corr[1, 0], M_corr[0, 0]))
    tx_corr, ty_corr = M_corr[0, 3], M_corr[1, 3]
    print(f"\nTop-floor correction:")
    print(f"  theta={theta_corr:.3f}°  tx={tx_corr:.3f} m  ty={ty_corr:.3f} m")
    print(f"\nRun:  uv run pipeline/cc_client.py apply")

    # ── Push updated PLYs into CC for visual check ────────────────────────────
    stair_out = OUT / "cc_staircase_aligned.ply"
    _write_ply(stair_out, stair_global)
    top_corr_pts = _apply4(top_pts, M_corr)
    top_corr_out = OUT / "cc_top_floor_corrected.ply"
    _write_ply(top_corr_out, top_corr_pts)

    r = _cc(f"""
for p in [r"{stair_out}", r"{top_corr_out}"]:
    ents = CC.loadFile(p)
    if ents: CC.redrawAll()
_result = "aligned clouds loaded"
""")
    if r["ok"]:
        print("Aligned clouds loaded in CC — check the viewer")


# ── Apply best transform to pipeline ─────────────────────────────────────────

def cmd_apply():
    if not BEST_T.exists():
        print("No saved transform — run 'register' first")
        return

    result = np.load(str(BEST_T), allow_pickle=True).item()
    M_corr  = result["M_corr"]
    fitness = result["fitness"]
    rmse    = result["rmse"]

    with open(OUT / "FINAL_transforms.pkl", "rb") as f:
        data = pickle.load(f)

    old_T = data["global_T"]["top_floor"].copy()
    new_T = M_corr @ old_T
    data["global_T"]["top_floor"] = new_T

    theta = np.degrees(np.arctan2(M_corr[1, 0], M_corr[0, 0]))
    print(f"Applying correction  θ={theta:.2f}°  tx={M_corr[0,3]:.3f}  ty={M_corr[1,3]:.3f}")
    print(f"  fitness={fitness:.4f}  rmse={rmse:.4f}")
    print(f"  top_floor: X {old_T[0,3]:.3f}→{new_T[0,3]:.3f}"
          f"  Y {old_T[1,3]:.3f}→{new_T[1,3]:.3f}"
          f"  Z {old_T[2,3]:.3f}→{new_T[2,3]:.3f}")

    with open(OUT / "FINAL_transforms.pkl", "wb") as f:
        pickle.dump(data, f)
    print("Saved FINAL_transforms.pkl")

    subprocess.run(
        ["uv", "run", str(Path(__file__).parent / "07_export_viewer.py")],
        cwd=str(Path(__file__).parent.parent),
    )
    print("Viewer GLBs updated — reload the browser.")


# ── Status ────────────────────────────────────────────────────────────────────

def cmd_status():
    with open(OUT / "FINAL_transforms.pkl", "rb") as f:
        data = pickle.load(f)
    print("global_T translations (Z-up):")
    for name, T in data["global_T"].items():
        theta = np.degrees(np.arctan2(T[1, 0], T[0, 0]))
        print(f"  {name:12s}  X={T[0,3]:7.3f}  Y={T[1,3]:7.3f}  Z={T[2,3]:7.3f}  θ={theta:.1f}°")
    if BEST_T.exists():
        r = np.load(str(BEST_T), allow_pickle=True).item()
        print(f"\nSaved registration: fitness={r['fitness']:.4f}  rmse={r['rmse']:.4f}")


# ── Interactive REPL ──────────────────────────────────────────────────────────

def cmd_repl():
    print("CloudCompare REPL — type Python code, send to CC.  'exit' to quit.")
    print("  _result = <expr>  to print a value back")
    print()
    lines = []
    while True:
        try:
            line = input("cc> " if not lines else "... ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line.strip() == "exit":
            break
        if line.strip() == "" and lines:
            code = "\n".join(lines)
            lines.clear()
            r = _cc(code)
            tag = "✓" if r["ok"] else "✗"
            print(f"{tag} {r['result']}")
        elif line.strip() == "" and not lines:
            pass
        else:
            lines.append(line)
            if not line.endswith("\\") and not line.endswith(":"):
                if lines:
                    code = "\n".join(lines)
                    lines.clear()
                    r = _cc(code)
                    tag = "✓" if r["ok"] else "✗"
                    print(f"{tag} {r['result']}")


# ── Entry point ───────────────────────────────────────────────────────────────

COMMANDS = {
    "export-ply": cmd_export_ply,
    "load":       cmd_load,
    "register":   cmd_register,
    "apply":      cmd_apply,
    "status":     cmd_status,
    "repl":       cmd_repl,
}

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "repl"
    if cmd not in COMMANDS:
        print(f"Unknown command '{cmd}'. Choose from: {', '.join(COMMANDS)}")
        sys.exit(1)
    COMMANDS[cmd]()
