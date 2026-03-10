#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import numpy as np

try:
    import trimesh
except ImportError as e:
    raise SystemExit("Please install: pip install trimesh pygltflib") from e


def make_transform(scale=1.0, rot_deg=(0.0, 0.0, 0.0), translate=(0.0, 0.0, 0.0), rot_order="xyz"):
    """
    Build a 4x4 transform matrix with:
      - uniform scale (or 3-tuple for non-uniform)
      - Euler rotation in degrees
      - translation
    rot_order: one of 'xyz','xzy','yxz','yzx','zxy','zyx'
    """
    # scale
    if isinstance(scale, (int, float)):
        sx, sy, sz = float(scale), float(scale), float(scale)
    else:
        sx, sy, sz = map(float, scale)

    # rotation (degrees -> radians)
    rx, ry, rz = np.deg2rad(rot_deg)

    def Rx(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0, 0],
                         [0, c,-s, 0],
                         [0, s, c, 0],
                         [0, 0, 0, 1]], dtype=np.float64)

    def Ry(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[ c, 0, s, 0],
                         [ 0, 1, 0, 0],
                         [-s, 0, c, 0],
                         [ 0, 0, 0, 1]], dtype=np.float64)

    def Rz(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c,-s, 0, 0],
                         [s, c, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=np.float64)

    Rmap = {"x": Rx(rx), "y": Ry(ry), "z": Rz(rz)}
    R = np.eye(4, dtype=np.float64)
    for ax in rot_order.lower():
        R = R @ Rmap[ax]

    S = np.array([[sx, 0,  0,  0],
                  [0,  sy, 0,  0],
                  [0,  0,  sz, 0],
                  [0,  0,  0,  1]], dtype=np.float64)

    tx, ty, tz = map(float, translate)
    T = np.array([[1, 0, 0, tx],
                  [0, 1, 0, ty],
                  [0, 0, 1, tz],
                  [0, 0, 0, 1]], dtype=np.float64)

    # apply scale -> rotation -> translation (TRS)
    M = T @ R @ S
    return M


def load_glb_as_scene(path: str):
    scene = trimesh.load(path, force="scene", process=False)
    if not isinstance(scene, trimesh.Scene):
        # if it's a single mesh, wrap it as scene
        sc = trimesh.Scene()
        sc.add_geometry(scene)
        scene = sc
    return scene


def apply_to_scene_geometry(scene: "trimesh.Scene", M: np.ndarray, about_center: bool = True):
    """
    Apply transform to every geometry in the scene.
    If about_center=True, rotate/scale about the scene bbox center (keeps it in place).
    """
    if about_center:
        bounds = scene.bounds  # (2,3)
        center = (bounds[0] + bounds[1]) * 0.5
        Tc = np.eye(4); Tc[:3, 3] = -center
        Tc_inv = np.eye(4); Tc_inv[:3, 3] = center
        M_use = Tc_inv @ M @ Tc
    else:
        M_use = M

    # apply transform to each geometry node transform
    # safest: bake into geometry vertices
    for name, geom in scene.geometry.items():
        if hasattr(geom, "apply_transform"):
            geom.apply_transform(M_use)

    return scene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_glb", required=True, help="input .glb/.gltf")
    ap.add_argument("--out_glb", required=True, help="output .glb")
    ap.add_argument("--scale", type=float, default=1.0, help="uniform scale, e.g. 0.01")
    ap.add_argument("--rx", type=float, default=0.0, help="rotation around X in degrees")
    ap.add_argument("--ry", type=float, default=0.0, help="rotation around Y in degrees")
    ap.add_argument("--rz", type=float, default=0.0, help="rotation around Z in degrees")
    ap.add_argument("--rot_order", type=str, default="xyz", help="euler order: xyz, zyx, ...")
    ap.add_argument("--tx", type=float, default=0.0, help="translation X")
    ap.add_argument("--ty", type=float, default=0.0, help="translation Y")
    ap.add_argument("--tz", type=float, default=0.0, help="translation Z")
    ap.add_argument("--about_center", type=int, default=1, help="1: rotate/scale about bbox center")
    args = ap.parse_args()

    scene = load_glb_as_scene(args.in_glb)

    M = make_transform(
        scale=args.scale,
        rot_deg=(args.rx, args.ry, args.rz),
        translate=(args.tx, args.ty, args.tz),
        rot_order=args.rot_order,
    )

    scene = apply_to_scene_geometry(scene, M, about_center=bool(args.about_center))

    # export as GLB
    data = scene.export(file_type="glb")
    with open(args.out_glb, "wb") as f:
        f.write(data)

    print(f"[OK] Saved: {args.out_glb}")


if __name__ == "__main__":
    main()