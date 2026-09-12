"""Generate OBJ mesh files for the visual fidelity pass (phase 8).

Creates irregular rock-like debris meshes and a sky dome, written as OBJ
files to ue_spike/Content/Meshes/. These are imported into UE at scene-
build time by pie_session.py.

Run standalone (no UE dependency):
    python ue_spike/Content/Python/generate_meshes.py

Meshes generated:
    debris_rock_0..4.obj  — 5 unique perturbed icospheres
    sky_dome.obj          — large inverted sphere for starfield backdrop
"""
import math
import os
import random

_HERE = os.path.dirname(os.path.abspath(__file__))
MESH_DIR = os.path.normpath(os.path.join(_HERE, "..", "Meshes"))


def _icosphere(subdivisions=2):
    """Create an icosphere by subdividing an icosahedron."""
    t = (1.0 + math.sqrt(5.0)) / 2.0
    verts = [
        (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
        (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
        (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1),
    ]
    # normalise to unit sphere
    verts = [_normalise(v) for v in verts]

    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]

    midpoint_cache = {}
    for _ in range(subdivisions):
        new_faces = []
        for tri in faces:
            a, b, c = tri
            ab = _midpoint(a, b, verts, midpoint_cache)
            bc = _midpoint(b, c, verts, midpoint_cache)
            ca = _midpoint(c, a, verts, midpoint_cache)
            new_faces.extend([
                (a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca),
            ])
        faces = new_faces
    return verts, faces


def _normalise(v):
    length = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    return (v[0] / length, v[1] / length, v[2] / length)


def _midpoint(i1, i2, verts, cache):
    key = (min(i1, i2), max(i1, i2))
    if key in cache:
        return cache[key]
    v1, v2 = verts[i1], verts[i2]
    mid = _normalise(((v1[0]+v2[0])/2, (v1[1]+v2[1])/2, (v1[2]+v2[2])/2))
    idx = len(verts)
    verts.append(mid)
    cache[key] = idx
    return idx


def generate_rock(seed, scale_range=(0.6, 1.4), noise_strength=0.3):
    """Perturb an icosphere to create irregular rock-like geometry."""
    rng = random.Random(seed)
    verts, faces = _icosphere(subdivisions=2)

    # non-uniform base scale
    sx = rng.uniform(*scale_range)
    sy = rng.uniform(*scale_range)
    sz = rng.uniform(*scale_range)

    perturbed = []
    for v in verts:
        r = 1.0 + rng.uniform(-noise_strength, noise_strength)
        perturbed.append((v[0] * sx * r, v[1] * sy * r, v[2] * sz * r))

    return perturbed, faces


def generate_sky_dome(radius=50000.0, subdivisions=3):
    """Large inverted sphere (normals pointing inward)."""
    verts, faces = _icosphere(subdivisions=subdivisions)
    scaled = [(v[0] * radius, v[1] * radius, v[2] * radius) for v in verts]
    # invert winding to flip normals inward
    inverted_faces = [(f[0], f[2], f[1]) for f in faces]
    return scaled, inverted_faces


def write_obj(verts, faces, filepath, comment=""):
    """Write a simple OBJ file (vertices + triangular faces)."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        if comment:
            f.write(f"# {comment}\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        # compute normals per-face for flat shading
        for tri in faces:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")


def generate_all():
    """Generate all mesh files for the visual fidelity pass."""
    os.makedirs(MESH_DIR, exist_ok=True)

    for i in range(5):
        verts, faces = generate_rock(
            seed=1000 + i * 137,
            scale_range=(0.5, 1.5),
            noise_strength=0.35,
        )
        path = os.path.join(MESH_DIR, f"debris_rock_{i}.obj")
        write_obj(verts, faces, path, f"debris rock {i} (perturbed icosphere, seed={1000+i*137})")
        print(f"wrote {path} ({len(verts)} verts, {len(faces)} faces)")

    verts, faces = generate_sky_dome(radius=50000.0, subdivisions=3)
    path = os.path.join(MESH_DIR, "sky_dome.obj")
    write_obj(verts, faces, path, "inverted sky dome for deep-space backdrop")
    print(f"wrote {path} ({len(verts)} verts, {len(faces)} faces)")


if __name__ == "__main__":
    generate_all()
