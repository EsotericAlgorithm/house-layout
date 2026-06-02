"""
Quick assessment of Polycam PLY point clouds:
- Point counts and bounding boxes
- Z-range overlap detection between floors
- Basic density and RGB coverage
"""

import struct
import numpy as np
import os

SCAN_DIR = os.path.join(os.path.dirname(__file__), "docs")
SCANS = {
    "main_floor": os.path.join(SCAN_DIR, "main_floor", "5_29_2026.ply"),
    "basement":   os.path.join(SCAN_DIR, "basement",   "5_29_2026.ply"),
    "top_floor":  os.path.join(SCAN_DIR, "top_floor",  "5_29_2026.ply"),
}


def parse_ply_header(f):
    """Returns (n_vertices, properties, header_end_byte)."""
    props = []
    n_verts = 0
    while True:
        line = f.readline().decode("ascii", errors="replace").strip()
        if line == "end_header":
            break
        if line.startswith("element vertex"):
            n_verts = int(line.split()[-1])
        if line.startswith("property"):
            parts = line.split()
            props.append((parts[1], parts[2]))  # (type, name)
    return n_verts, props


def dtype_from_props(props):
    type_map = {
        "double": "f8", "float": "f4",
        "uchar": "u1", "uint8": "u1",
        "int": "i4", "uint": "u4",
        "short": "i2", "ushort": "u2",
    }
    return np.dtype([(name, type_map[t]) for t, name in props])


def load_ply(path):
    with open(path, "rb") as f:
        n_verts, props = parse_ply_header(f)
        dt = dtype_from_props(props)
        data = np.frombuffer(f.read(n_verts * dt.itemsize), dtype=dt)
    return data


def analyze(name, data):
    x, y, z = data["x"], data["y"], data["z"]
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Points      : {len(data):,}")
    print(f"  X range     : {x.min():.3f} → {x.max():.3f}  (span {x.max()-x.min():.2f} m)")
    print(f"  Y range     : {y.min():.3f} → {y.max():.3f}  (span {y.max()-y.min():.2f} m)")
    print(f"  Z range     : {z.min():.3f} → {z.max():.3f}  (span {z.max()-z.min():.2f} m)")
    if "red" in data.dtype.names:
        has_color = np.any(data["red"] > 0) or np.any(data["green"] > 0) or np.any(data["blue"] > 0)
        avg_r = data["red"].mean()
        avg_g = data["green"].mean()
        avg_b = data["blue"].mean()
        print(f"  Has color   : {has_color}  (avg RGB: {avg_r:.0f}, {avg_g:.0f}, {avg_b:.0f})")
    return {"name": name, "x": x, "y": y, "z": z, "n": len(data)}


def check_overlap(a, b, z_margin=0.1):
    """Reports Z-axis overlap between two scans."""
    a_zmin, a_zmax = a["z"].min(), a["z"].max()
    b_zmin, b_zmax = b["z"].min(), b["z"].max()
    overlap_lo = max(a_zmin, b_zmin)
    overlap_hi = min(a_zmax, b_zmax)
    if overlap_hi > overlap_lo:
        span = overlap_hi - overlap_lo
        print(f"\n  Z overlap {a['name']} <-> {b['name']}: {overlap_lo:.3f} → {overlap_hi:.3f} ({span:.3f} m)")
        a_in = np.sum((a["z"] >= overlap_lo - z_margin) & (a["z"] <= overlap_hi + z_margin))
        b_in = np.sum((b["z"] >= overlap_lo - z_margin) & (b["z"] <= overlap_hi + z_margin))
        print(f"    Points in overlap zone: {a['name']}={a_in:,}  {b['name']}={b_in:,}")
    else:
        gap = overlap_lo - overlap_hi
        print(f"\n  No Z overlap {a['name']} <-> {b['name']} (gap: {gap:.3f} m)")


print("Loading scans...")
scans = {}
for name, path in SCANS.items():
    print(f"  {name}: {path}")
    data = load_ply(path)
    scans[name] = analyze(name, data)

print("\n\n--- OVERLAP ANALYSIS ---")
names = list(scans.values())
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        check_overlap(names[i], names[j])

print("\n\n--- SUMMARY ---")
print("Z ranges (use to infer floor order):")
for s in scans.values():
    print(f"  {s['name']}: z {s['z'].min():.2f} → {s['z'].max():.2f}")
