import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import imageio.v3 as iio
from PIL import Image
from pathlib import Path
import warnings
import os
import sys
import json
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

from lang_sam import LangSAM

import trimesh
import pyrender
from pytorch3d.renderer import FoVPerspectiveCameras
from pytorch3d.transforms import axis_angle_to_matrix

import matplotlib
matplotlib.use("Agg")  # Headless-safe
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
from tqdm import tqdm

if "DISPLAY" not in os.environ:
    os.environ["PYOPENGL_PLATFORM"] = "egl"


def ensure_dir(p: str) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path

def normalize_langsam_results(results, fallback_text="object"):
    if isinstance(results, list):
        pack = results[0]
    elif isinstance(results, dict):
        pack = results
    elif isinstance(results, tuple) and len(results) == 4:
        masks, boxes, phrases, logits = results
        pack = {"masks": masks, "boxes": boxes, "phrases": phrases, "logits": logits}
    else:
        raise TypeError(f"Unsupported results type: {type(results)}")
    
    masks, boxes, phrases, logits = pack["masks"], pack["boxes"], pack.get("phrases"), pack.get("logits")
    if hasattr(masks, "cpu"): masks = masks.cpu().numpy()
    if hasattr(boxes, "cpu"): boxes = boxes.cpu().numpy()
    if logits is not None and hasattr(logits, "cpu"): logits = logits.cpu().numpy()
    
    if masks.ndim == 2:
        masks = masks[None, ...]
    elif masks.ndim != 3:
        raise ValueError(f"Unexpected masks shape: {masks.shape}")
        
    N = masks.shape[0]
    if phrases is None:
        phrases = [fallback_text] * N
    if logits is not None and np.ndim(logits) == 0:
        logits = np.array([float(logits)])
    if logits is not None and len(logits) != N:
        logits = None
        
    return {"masks": masks, "boxes": boxes, "phrases": list(phrases), "logits": logits}

def choose_mask(bin_masks):
    if not bin_masks:
        return None, -1
    areas = [int(m.sum()) for m in bin_masks]
    k = int(np.argmax(areas))
    return bin_masks[k], k

def to_numpy_f32(x: torch.Tensor) -> np.ndarray:
    return x.detach().to(dtype=torch.float32).cpu().numpy()

def _poisson_sample_2d(points2d: np.ndarray, max_points: int, img_w: int, img_h: int) -> np.ndarray:
    K = points2d.shape[0]
    if K == 0 or max_points <= 0:
        return np.zeros((0,), dtype=np.int64)
    if K <= max_points:
        return np.arange(K, dtype=np.int64)

    area = float(img_w * img_h)
    base_r = np.sqrt(area / (max_points * np.pi))
    min_d2 = base_r * base_r / 5

    indices = np.random.permutation(K)
    selected = []

    for idx in indices:
        p = points2d[idx]
        if np.any(np.isnan(p)): continue
        ok = True
        for j in selected:
            q = points2d[j]
            if (p[0]-q[0])**2 + (p[1]-q[1])**2 < min_d2:
                ok = False
                break
        if ok:
            selected.append(idx)
            if len(selected) >= max_points: break

    if not selected:
        selected = [indices[0]]
    return np.array(selected, dtype=np.int64)

def extract_joints_from_glb(glb_path):

    try:
        scene = trimesh.load(glb_path, force='scene')
        joints = []
            
        for node_name in scene.graph.nodes:
            if node_name not in scene.geometry:
                transform, _ = scene.graph.get(node_name)
                pos = transform[:3, 3]
                if np.linalg.norm(pos) > 1e-5:
                    joints.append(pos)
                    
        if len(joints) > 5: 
            joints_np = np.array(joints, dtype=np.float32)
            return joints_np
            
    except Exception as e:
        print(f"error extracting joints from GLB: {e}")
        
    return None

# --------------------------- (Pyrender) ---------------------------
def render_mesh(mesh_path, num_blobs, cam_t, cam_r_pyr, yfov_deg, image_size, device, explicit_blobs=None):
    H, W = image_size
    
    if explicit_blobs is None and mesh_path.lower().endswith('.glb'):
        extracted_joints = extract_joints_from_glb(mesh_path)
        if extracted_joints is not None:
            explicit_blobs = extracted_joints
            
    mesh_tm = trimesh.load(mesh_path, force='mesh')
    
    if isinstance(mesh_tm, trimesh.Scene):
        mesh_tm = trimesh.util.concatenate(tuple(mesh_tm.geometry.values()))

    if explicit_blobs is not None:
        blobs_3d_np_all = explicit_blobs.astype(np.float32)
        use_explicit = True
    else:
        use_explicit = False
        oversample_factor = 5
        num_sample = max(num_blobs * oversample_factor, num_blobs)
        blobs_3d_np_all, _ = trimesh.sample.sample_surface(mesh_tm, num_sample)

    num_points = blobs_3d_np_all.shape[0]

    pyrender_mesh = pyrender.Mesh.from_trimesh(mesh_tm, smooth=True)
    scene = pyrender.Scene(ambient_light=[10.0, 10.0, 10.0], bg_color=[0.8, 0.8, 0.8, 0])
    scene.add(pyrender_mesh)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=50.0)
    scene.add(light, pose=np.eye(4))

    cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(yfov_deg), aspectRatio=W / H)
    
    eye = np.array(cam_t, dtype=np.float32)
    target = np.array([0.0, 30.0, 0.0], dtype=np.float32)

    forward = target - eye
    forward = forward / (np.linalg.norm(forward) + 1e-8)

    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(np.dot(forward, up)) > 0.99:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-8)

    true_up = np.cross(right, forward)
    true_up = true_up / (np.linalg.norm(true_up) + 1e-8)

    R_cam2world = np.stack([right, true_up, -forward], axis=1)

    cam_pose = np.eye(4, dtype=np.float32)
    cam_pose[:3, :3] = R_cam2world
    cam_pose[:3, 3] = eye

    scene.add(cam, pose=cam_pose)
    # R_pyr, _ = cv2.Rodrigues(np.array(cam_r_pyr, dtype=np.float32))
    # cam_pose = np.eye(4)
    # cam_pose[:3, :3] = R_pyr
    # cam_pose[:3, 3] = cam_t
    # scene.add(cam, pose=cam_pose)
    
    cam_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=20.0)
    scene.add(cam_light, pose=cam_pose)

    r = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
    rendered_image, _ = r.render(scene)
    r.delete()

    view_matrix = np.linalg.inv(cam_pose)
    proj_matrix = cam.get_projection_matrix(width=W, height=H)

    blobs_3d_hom = np.hstack([blobs_3d_np_all, np.ones((num_points, 1))])
    points_clip = (proj_matrix @ view_matrix @ blobs_3d_hom.T).T

    w = points_clip[:, 3]
    points_ndc = points_clip[:, :3] / w[:, None]

    x_pix = (points_ndc[:, 0] + 1) * 0.5 * W
    y_pix = (1 - points_ndc[:, 1]) * 0.5 * H
    blobs_2d_np_all = np.stack([x_pix, y_pix], axis=1)

    visible_mask_all = (w > 0) & (points_ndc[:, 0] >= -1) & (points_ndc[:, 0] <= 1) & \
                       (points_ndc[:, 1] >= -1) & (points_ndc[:, 1] <= 1)
    visible_idx_all = np.where(visible_mask_all)[0]

    if use_explicit:
        blobs_3d_np, blobs_2d_np, visible_mask_np = blobs_3d_np_all, blobs_2d_np_all, np.ones(blobs_3d_np_all.shape[0], dtype=bool)
    else:
        if visible_idx_all.size == 0:
            blobs_3d_np, blobs_2d_np, visible_mask_np = blobs_3d_np_all, blobs_2d_np_all, visible_mask_all
        else:
            pts2d_vis = blobs_2d_np_all[visible_idx_all]
            max_points = min(num_blobs, pts2d_vis.shape[0])
            sel_rel = _poisson_sample_2d(pts2d_vis, max_points, W, H)
            sel_idx = visible_idx_all[sel_rel]
            blobs_3d_np, blobs_2d_np = blobs_3d_np_all[sel_idx], blobs_2d_np_all[sel_idx]
            visible_mask_np = np.ones(blobs_3d_np.shape[0], dtype=bool)

    return (rendered_image, 
        torch.from_numpy(blobs_3d_np).float().to(device), 
        torch.from_numpy(blobs_2d_np).float().to(device), 
        torch.from_numpy(visible_mask_np).to(device),
        cam_pose)


@torch.no_grad()
def auto_match_2d_3d(img_vid, pts_vid, img_ren, pts_ren, device, w_spatial=1.5, thresh=2.5):
    print("  -> 載入 DINOv2 模型進行特徵匹配...")
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device).eval()

    def extract_dense_features(img_np):
        H, W = img_np.shape[:2]
        nH, nW = (H // 14) * 14, (W // 14) * 14
        t = torch.from_numpy(img_np.copy()).float().to(device).permute(2,0,1).unsqueeze(0) / 255.0 
        t = F.interpolate(t, size=(nH, nW), mode='bilinear', align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
        t = (t - mean) / std
        feats = dino.forward_features(t)['x_norm_patchtokens']
        C = feats.shape[-1]
        feats = feats.reshape(1, nH//14, nW//14, C).permute(0, 3, 1, 2)
        return F.interpolate(feats, size=(H, W), mode='bilinear', align_corners=False)[0]

    f_map_vid = extract_dense_features(img_vid)
    f_map_ren = extract_dense_features(img_ren)

    def sample_features(f_map, pts):
        pts_c = pts.long().clamp(min=0)
        pts_c[:, 0] = pts_c[:, 0].clamp(max=f_map.shape[2]-1)
        pts_c[:, 1] = pts_c[:, 1].clamp(max=f_map.shape[1]-1)
        return f_map[:, pts_c[:, 1], pts_c[:, 0]].T

    f_vid = F.normalize(sample_features(f_map_vid, pts_vid), dim=1)
    f_ren = F.normalize(sample_features(f_map_ren, pts_ren), dim=1)

    cost_feat = 1.0 - torch.mm(f_vid, f_ren.T) 

    def norm_spatial(p):
        p_min, p_max = p.min(dim=0)[0], p.max(dim=0)[0]
        return (p - p_min) / (p_max - p_min + 1e-6)

    cost_spat = torch.cdist(norm_spatial(pts_vid), norm_spatial(pts_ren)) 

    total_cost = (cost_feat + w_spatial * cost_spat).cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(total_cost)

    matches = []
    for r, c in zip(row_ind, col_ind):
        if total_cost[r, c] < thresh:
            matches.append((r, c))

    return matches, total_cost


# --------------------------- 2D blob spacing ---------------------------
def filter_blobs_by_min_2d_distance(blobs2d: torch.Tensor, min_dist: float) -> torch.Tensor:
    if min_dist <= 0.0 or blobs2d.shape[0] == 0:
        return torch.arange(blobs2d.shape[0], device=blobs2d.device, dtype=torch.long)
    min_d2 = float(min_dist * min_dist)
    pts = blobs2d.detach().cpu().numpy()
    kept = []
    for i, (x, y) in enumerate(pts):
        ok = True
        for j in kept:
            if (x - pts[j, 0])**2 + (y - pts[j, 1])**2 < min_d2:
                ok = False
                break
        if ok: kept.append(i)
    if not kept: kept = [0]
    return torch.tensor(kept, dtype=torch.long, device=blobs2d.device)


def compute_shape_loss(p_gt: torch.Tensor, p_pred: torch.Tensor):
    p_gt, p_pred = p_gt.to(torch.float32), p_pred.to(torch.float32)
    if p_gt.shape[0] < 2: return torch.zeros([], device=p_gt.device, dtype=torch.float32)
    pg, pp = p_gt - p_gt.mean(dim=0, keepdim=True), p_pred - p_pred.mean(dim=0, keepdim=True)
    return ((pg / pg.norm(dim=1).mean().clamp(min=1e-8)) - (pp / pp.norm(dim=1).mean().clamp(min=1e-8))).pow(2).mean()

def monotonic_order_loss(p_gt: torch.Tensor, p_pred: torch.Tensor, num_pairs: int = 512):
    M = p_gt.shape[0]
    if M < 2: return torch.tensor(0.0, device=p_gt.device)
    i_idx = torch.randint(0, M, (num_pairs,), device=p_gt.device)
    j_idx = torch.randint(0, M, (num_pairs,), device=p_gt.device)
    mask = i_idx != j_idx
    i_idx, j_idx = i_idx[mask], j_idx[mask]
    if i_idx.numel() == 0: return torch.tensor(0.0, device=p_gt.device)

    dx_gt = p_gt[i_idx, 0] - p_gt[j_idx, 0]
    dx_pr = p_pred[i_idx, 0] - p_pred[j_idx, 0]
    dy_gt = p_gt[i_idx, 1] - p_gt[j_idx, 1]
    dy_pr = p_pred[i_idx, 1] - p_pred[j_idx, 1]

    loss_x = torch.relu(-(dx_gt * dx_pr)).mean()
    loss_y = torch.relu(-(dy_gt * dy_pr)).mean()
    return (loss_x + loss_y) * 0.5

@torch.no_grad()
def precompute_arap_graph(P_3D_rest, k=10):
    N = P_3D_rest.shape[0]
    if N <= 1:
        return torch.empty((0, 2), dtype=torch.long, device=P_3D_rest.device), torch.empty((0,), dtype=torch.float32, device=P_3D_rest.device)
    k = min(k, max(N - 1, 1))
    d = torch.cdist(P_3D_rest, P_3D_rest)
    _, idx = torch.topk(d, k + 1, dim=1, largest=False)
    src = torch.arange(N, device=P_3D_rest.device)[:, None].repeat(1, k).reshape(-1)
    dst = idx[:, 1:].reshape(-1)
    edges = torch.stack([src, dst], dim=1)
    dist0 = (P_3D_rest[edges[:, 0]] - P_3D_rest[edges[:, 1]]).norm(dim=1)
    return edges.detach(), dist0.detach()


def _pinhole_K_from_yfov(yfov_deg: float, H: int, W: int):
    yfov = np.deg2rad(float(yfov_deg))
    fy = 0.5 * H / np.tan(0.5 * yfov)
    fx = 0.5 * W / np.tan(0.5 * yfov)
    cx = 0.5 * W
    cy = 0.5 * H
    return fx, fy, cx, cy

@torch.no_grad()
@torch.no_grad()
def compute_3d_motion_from_2d_tracks(
    tracks_np, matched_rows, matched_cols, P_final,
    cam_pose=None, yfov_deg=60.0, image_size=None,
    plane="yz", flip_image_y=True
):
    device = P_final.device
    T = tracks_np.shape[0]
    K = P_final.shape[0]

    t0_2d = torch.from_numpy(tracks_np[0]).float().to(device)
    p2d_t0 = t0_2d[matched_rows]
    p3d_t0 = P_final[matched_cols]
    
    if cam_pose is not None and image_size is not None:
        H, W = int(image_size[0]), int(image_size[1])
        fx, fy, cx, cy = _pinhole_K_from_yfov(yfov_deg, H, W)

        cam_pose_np = np.array(cam_pose, dtype=np.float32)
        world2cam_np = np.linalg.inv(cam_pose_np)

        t0_2d = torch.from_numpy(tracks_np[0]).float().to(device)
        p2d_t0 = t0_2d[matched_rows]              # (M,2) matched tracks in pixel
        p3d_t0 = P_final[matched_cols]           # (M,3) matched blobs in world

        # world -> cam(OpenGL camera coords; forward is -Z)
        ones = torch.ones((p3d_t0.shape[0], 1), dtype=p3d_t0.dtype, device=device)
        p3d_h = torch.cat([p3d_t0, ones], dim=1)  # (M,4)

        world2cam = torch.from_numpy(world2cam_np).to(device=device, dtype=p3d_t0.dtype)  # (4,4)
        p_cam0_h = (world2cam @ p3d_h.t()).t()  # (M,4)
        p_cam0 = p_cam0_h[:, :3]                # (M,3) in OpenGL cam coords

        # OpenGL cam coords: camera looks along -Z, so "positive depth" = -z
        z_fwd = (-p_cam0[:, 2]).clamp(min=1e-6)  # (M,)

        def pix_to_norm(u, v):
            x_n = (u - cx) / fx
            y_n = (cy - v) / fy
            return x_n, y_n

        u0 = p2d_t0[:, 0]
        v0 = p2d_t0[:, 1]
        x0_n, y0_n = pix_to_norm(u0, v0)

        # pinhole cam coords: z forward positive
        X0 = x0_n * z_fwd
        Y0 = y0_n * z_fwd
        Z0 = z_fwd

        # 轉回 OpenGL cam coords：z_gl = -Z0
        p_cam0_frompix = torch.stack([X0, Y0, -Z0], dim=1)  # (M,3)

        R_cam2world = torch.from_numpy(cam_pose_np[:3, :3]).to(device=device, dtype=p3d_t0.dtype)

        D_dyn_matched = torch.zeros((T, len(matched_cols), 3), dtype=P_final.dtype, device=device)

        for t in range(T):
            pt = torch.from_numpy(tracks_np[t]).float().to(device)[matched_rows]
            ut = pt[:, 0]
            vt = pt[:, 1]
            xt_n, yt_n = pix_to_norm(ut, vt)

            Xt = xt_n * z_fwd
            Yt = yt_n * z_fwd
            Zt = z_fwd
            p_camt = torch.stack([Xt, Yt, -Zt], dim=1)  # OpenGL cam coords

            d_cam = p_camt - p_cam0_frompix           # (M,3) in cam coords
            d_world = (R_cam2world @ d_cam.t()).t()   # (M,3) rotate into world

            D_dyn_matched[t] = d_world

        dist = torch.cdist(P_final, P_final[matched_cols])
        weights = 1.0 / (dist + 1e-6)
        weights = weights / weights.sum(dim=1, keepdim=True)

        D_dyn_all = torch.zeros((T, K, 3), dtype=P_final.dtype, device=device)
        for t in range(T):
            D_dyn_all[t] = torch.mm(weights, D_dyn_matched[t])

        P_dyn = P_final.unsqueeze(0) + D_dyn_all

        k_scale = 1.0
        return P_dyn, D_dyn_all, k_scale

    axes_map = {"yz": (1, 2), "xy": (0, 1), "xz": (0, 2)}
    ax0, ax1 = axes_map[plane.lower()]

    rad2d = torch.median((p2d_t0 - p2d_t0.mean(0, keepdim=True)).norm(dim=1)).item()
    plane_pts = p3d_t0[:, (ax0, ax1)]
    rad3d = torch.median((plane_pts - plane_pts.mean(0, keepdim=True)).norm(dim=1)).item()
    k_scale = float(rad3d / max(rad2d, 1e-8))

    D_dyn_matched = torch.zeros((T, len(matched_cols), 3), dtype=P_final.dtype, device=device)
    for t in range(T):
        d2d_m = (torch.from_numpy(tracks_np[t]).float().to(device) - t0_2d)[matched_rows]
        D_dyn_matched[t, :, ax0] = d2d_m[:, 0] * k_scale
        D_dyn_matched[t, :, ax1] = d2d_m[:, 1] * (-k_scale if flip_image_y else k_scale)

    dist = torch.cdist(P_final, P_final[matched_cols])
    weights = 1.0 / (dist + 1e-6)
    weights = weights / weights.sum(dim=1, keepdim=True)

    D_dyn_all = torch.zeros((T, K, 3), dtype=P_final.dtype, device=device)
    for t in range(T):
        D_dyn_all[t] = torch.mm(weights, D_dyn_matched[t])

    P_dyn = P_final.unsqueeze(0) + D_dyn_all
    
    return P_dyn, D_dyn_all, k_scale

def _set_axes_equal(ax):
    extents = np.array([getattr(ax, f'get_{dim}lim')() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, f'set_{dim}lim')(ctr - r, ctr + r)

def visualize_correspondences(img1, pts1, img2, pts2, corres, scores, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(img1); ax1.axis('off'); ax1.set_title("2D Tracks")
    ax2.imshow(img2); ax2.axis('off'); ax2.set_title("Rendered Blobs")

    valid = np.where(corres >= 0)[0]
    colors = plt.get_cmap('hsv', max(len(valid), 1))
    
    if hasattr(pts1, "cpu"): pts1 = pts1.cpu().numpy()
    if hasattr(pts2, "cpu"): pts2 = pts2.cpu().numpy()

    for idx, i in enumerate(valid):
        c = colors(idx / max(len(valid), 1))
        ax1.scatter(*pts1[i], c=[c], s=80, edgecolors='w')
        ax1.text(pts1[i,0]+5, pts1[i,1]+5, str(i), color='w', bbox=dict(facecolor=c, alpha=0.7, pad=1))
        
        ax2.scatter(*pts2[corres[i]], c=[c], s=80, edgecolors='w')
        ax2.text(pts2[corres[i],0]+5, pts2[corres[i],1]+5, str(i), color='w', bbox=dict(facecolor=c, alpha=0.7, pad=1))

    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

def visualize_2d_3d_side_by_side_video(all_frames, tracks_np, P_dyn_tensor, mesh_path, save_path, fps=30):
    T, H, W, _ = all_frames.shape
    P_dyn = P_dyn_tensor.cpu().numpy()
    
    try:
        scene = trimesh.load(mesh_path, force='scene')
        if isinstance(scene, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(scene.geometry.values()))
        else:
            mesh = scene
        V, F = mesh.vertices, mesh.faces
        mesh_poly = Poly3DCollection(V[F], facecolor=(0.8, 0.8, 0.8, 0.1), alpha=0.1)
    except: mesh_poly, V = None, None

    fig = plt.figure(figsize=(20, 8))
    ax1 = fig.add_subplot(121); im = ax1.imshow(all_frames[0])
    sc1 = ax1.scatter(*tracks_np[0].T, c=[plt.get_cmap('hsv', len(tracks_np[0]))(i) for i in range(len(tracks_np[0]))])
    ax1.set(xlim=(0,W), ylim=(H,0)); ax1.axis('off')

    ax2 = fig.add_subplot(122, projection='3d')
    if mesh_poly: ax2.add_collection3d(mesh_poly)
    sc2 = ax2.scatter(*P_dyn[0].T, s=15, c='red')
    
    mins, maxs = P_dyn.reshape(-1,3).min(0), P_dyn.reshape(-1,3).max(0)
    c = (mins + maxs)/2; r = (maxs - mins).max()/2 * 1.1
    ax2.set(xlim=(c[0]-r, c[0]+r), ylim=(c[1]-r, c[1]+r), zlim=(c[2]-r, c[2]+r))
    ax2.view_init(90, -90)

    def update(t):
        im.set_data(all_frames[t]); sc1.set_offsets(tracks_np[t])
        sc2._offsets3d = tuple(P_dyn[t].T)
        return im, sc1, sc2

    animation.FuncAnimation(fig, update, frames=T, interval=1000/fps).save(str(save_path), writer='ffmpeg', fps=fps)
    plt.close()

def extract_video_tracks(video_path, text_prompt, grid_size, erode, device, outdir: Path):
    frames = iio.imread(video_path, plugin="FFMPEG")
    H, W = frames[0].shape[:2]

    pack = normalize_langsam_results(LangSAM().predict([Image.fromarray(frames[0]).convert("RGB")], [text_prompt]))
    mask0, _ = choose_mask([(pack["masks"][i] > 0.5).astype(np.uint8) * 255 for i in range(pack["masks"].shape[0])])
    if mask0 is None: raise RuntimeError("Lang-SAM 未產生遮罩。")
    mask0 = cv2.resize(mask0, (W, H), interpolation=cv2.INTER_NEAREST)

    with torch.no_grad():
        pred_tracks, pred_vis = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)(
            torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device), grid_size=grid_size)

    tracks, vis = pred_tracks[0].cpu().numpy(), pred_vis[0].cpu().numpy()[..., 0] if pred_vis.dim() > 3 else pred_vis[0].cpu().numpy()
    m_eroded = cv2.erode(mask0, np.ones((2*erode+1, 2*erode+1))) if erode > 0 else mask0
    
    keep = [i for i, (x, y) in enumerate(tracks[0]) if 0 <= int(y) < H and 0 <= int(x) < W and m_eroded[int(y), int(x)] > 0 and vis[0, i] > 0.5]
    if not keep: raise RuntimeError("遮罩內無有效軌跡點。")
    
    cv2.imwrite(str(outdir / "frame0_mask.png"), mask0)
    return tracks[:, keep, :], frames, mask0


# --------------------------- 主流程 ---------------------------
def main():
    parser = argparse.ArgumentParser(description="2D→3D 自動對應與輕量最佳化")

    # Video args
    parser.add_argument("--video", type=str, required=True, help="輸入影片路徑")
    parser.add_argument("--text", type=str, default="woody", help="LangSAM 的文本提示")
    parser.add_argument("--grid_size", type=int, default=50, help="CoTracker 的採樣網格大小")
    parser.add_argument("--erode", type=int, default=3, help="侵蝕遮罩的像素寬度")
    parser.add_argument("--fps", type=int, default=30, help="輸出影片的幀率 (FPS)")

    # Mesh args
    parser.add_argument("--mesh", type=str, required=True, help="輸入 3D Mesh 路徑 (.obj, .glb)")
    parser.add_argument("--num_blobs", type=int, default=30, help="在 Mesh 表面採樣的 3D 點數量")
    parser.add_argument("--yfov_deg", type=float, default=60.0, help="Pyrender 相機 Y-FoV")
    parser.add_argument("--init_blobs_path", type=str, default=None, help="從 .npy 讀取初始 3D Blobs (若無則自動探測 RigAnything 骨架)")

    # [NEW] Camera parameters from JSON
    parser.add_argument("--cam_json", type=str, default=None, help="包含相機設定的 JSON 檔案路徑")

    # Matching weights
    parser.add_argument("--w_xy", type=float, default=1.0)
    parser.add_argument("--w_rank", type=float, default=1.0)
    parser.add_argument("--w_ang", type=float, default=0.5)
    parser.add_argument("--w_img", type=float, default=0.0)
    parser.add_argument("--patch_radius", type=int, default=5)

    # Optimization args
    parser.add_argument("--optim_steps", type=int, default=1000)
    parser.add_argument("--optim_lr", type=float, default=1e-3)
    parser.add_argument("--w_shape", type=float, default=1.0)
    parser.add_argument("--w_order", type=float, default=0.2)
    parser.add_argument("--w_arap", type=float, default=1.0)
    parser.add_argument("--arap_k", type=int, default=10)
    parser.add_argument("--w_delta_l2", type=float, default=1e-4)
    parser.add_argument("--w_delta_l1", type=float, default=0.0)
    parser.add_argument("--disable_optim", action="store_true", help="關閉 3D 最佳化")

    # Misc args
    parser.add_argument("--min_blob_2d_dist", type=float, default=0.0)
    parser.add_argument("--outdir", type=str, default="./output/1_point2blobs", help="輸出目錄")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dash_port", type=int, default=8050) 

    args = parser.parse_args()
    outdir = ensure_dir(args.outdir)
    device = torch.device(args.device)

    # === 解析 Camera JSON ===
    cam_t = [0.0, 1.8, 5.0]
    cam_r_p3d = [0.0, np.pi, 0.0]
    cam_r_pyr = [0.0, 0.0, 0.0]

    if args.cam_json and os.path.exists(args.cam_json):
        print(f"讀取相機設定檔: {args.cam_json}")
        with open(args.cam_json, 'r') as f:
            cam_data = json.load(f)
            cam_t = cam_data.get("cam_t", cam_t)
            cam_r_p3d = cam_data.get("cam_r_p3d", cam_r_p3d)
            cam_r_pyr = cam_data.get("cam_r_pyr", cam_r_pyr)
    else:
        if args.cam_json:
            print(f"Warning: cam_json not found at {args.cam_json}. Using default camera parameters.")
        else:
            print("no cam_json provided. Using default camera parameters.")

    tracks_np, all_frames, mask0 = extract_video_tracks(args.video, args.text, args.grid_size, args.erode, device, outdir)
    frame0_rgb = all_frames[0]
    points_2d_t0 = torch.from_numpy(tracks_np[0]).float().to(device)
    H, W = frame0_rgb.shape[:2]

    explicit_blobs = np.load(args.init_blobs_path) if args.init_blobs_path else None
    
    rendered_image, blobs_3d_all, blobs_2d_all, vis_mask, cam_pose = render_mesh(
        args.mesh, args.num_blobs, cam_t, cam_r_pyr, args.yfov_deg, (H, W), device, explicit_blobs=explicit_blobs
    )
    vis_idx = torch.where(vis_mask)[0]
    
    P_rest = blobs_3d_all.clone().detach() 
    N = P_rest.shape[0]

    blobs_3d_vis, blobs_2d_vis = blobs_3d_all[vis_idx], blobs_2d_all[vis_idx]

    if args.min_blob_2d_dist > 0.0 and explicit_blobs is None:
        kept_idx = filter_blobs_by_min_2d_distance(blobs_2d_vis, args.min_blob_2d_dist)
        blobs_3d_vis, blobs_2d_vis = blobs_3d_vis[kept_idx], blobs_2d_vis[kept_idx]
        vis_idx = vis_idx[kept_idx]

    matches_list, cost_matrix = auto_match_2d_3d(
        frame0_rgb, points_2d_t0, rendered_image, blobs_2d_vis, device
    )

    if not matches_list:
        sys.exit(1)

    
    M = points_2d_t0.shape[0]
    row_to_col_vis = np.full(M, -1, dtype=np.int64)
    row_cost = np.full(M, np.inf, dtype=np.float64)

    for l_idx, r_idx in matches_list:
        row_to_col_vis[l_idx] = r_idx
        row_cost[l_idx] = cost_matrix[l_idx, r_idx]

    matched_rows_np = np.where(row_to_col_vis >= 0)[0]
    matched_cols_vis_np = row_to_col_vis[matched_rows_np]
    
    matched_cols_global_np = vis_idx[matched_cols_vis_np].cpu().numpy()
    matched_cols_global_t = torch.from_numpy(matched_cols_global_np).long().to(device)

    p_gt_matched = points_2d_t0[matched_rows_np]

    scores = np.zeros_like(row_cost, dtype=np.float32)
    good = np.isfinite(row_cost)
    scores[good] = 1.0 / (1.0 + row_cost[good]) 
    visualize_correspondences(frame0_rgb, points_2d_t0, rendered_image, blobs_2d_vis, row_to_col_vis, scores, outdir / "auto_correspondences.jpg")

    R_axis = torch.nn.Parameter(torch.tensor([cam_r_p3d], dtype=torch.float32, device=device))
    T_vec = torch.nn.Parameter(torch.tensor([cam_t], dtype=torch.float32, device=device))
    cam = FoVPerspectiveCameras(device=device, fov=args.yfov_deg, aspect_ratio=W/H, R=axis_angle_to_matrix(R_axis), T=T_vec)
    image_size_tensor = torch.tensor([[H, W]], device=device, dtype=torch.float32)

    if args.disable_optim or args.optim_steps <= 0:
        P_final = P_rest.clone().detach()
    else:
        Delta = torch.nn.Parameter(torch.zeros_like(P_rest))
        arap_edges, arap_dist0 = precompute_arap_graph(P_rest, k=args.arap_k)
        optimizer = torch.optim.Adam([{'params': [Delta], 'lr': args.optim_lr}, {'params': [R_axis, T_vec], 'lr': args.optim_lr*0.1}])

        pbar = tqdm(range(args.optim_steps))
        for step in pbar:
            optimizer.zero_grad()
            P_curr = P_rest + Delta
            cam.R, cam.T = axis_angle_to_matrix(R_axis), T_vec
            p_pred = cam.transform_points_screen(P_curr, image_size=image_size_tensor)[..., :2]

            p_pred_matched = p_pred[matched_cols_global_t]
            loss_shape = compute_shape_loss(p_gt_matched, p_pred_matched) * args.w_shape
            loss_order = monotonic_order_loss(p_gt_matched, p_pred_matched) * args.w_order
            
            loss_arap = torch.tensor(0.0, device=device)
            if arap_edges.numel() > 0:
                loss_arap = ((P_curr[arap_edges[:, 0]] - P_curr[arap_edges[:, 1]]).norm(dim=1) - arap_dist0).pow(2).mean() * args.w_arap

            loss = loss_shape + loss_order + loss_arap + (Delta.pow(2).mean() * args.w_delta_l2) + (Delta.abs().mean() * args.w_delta_l1)
            loss.backward()
            if Delta.grad is not None: Delta.grad[:, 2] = 0.0
            optimizer.step()
            if step % 100 == 0: pbar.set_postfix(loss=loss.item())

        P_final = (P_rest + Delta).detach()

    P_dyn, D_dyn, k_scale = compute_3d_motion_from_2d_tracks(
        tracks_np,
        torch.from_numpy(matched_rows_np).long().to(device),
        matched_cols_global_t,
        P_final,
        cam_pose=cam_pose,
        yfov_deg=args.yfov_deg,
        image_size=(H, W),
        plane="xy",
        flip_image_y=True
    )
    torch.save(P_dyn, outdir / "3d_motion_blobs.pth")
    np.save(outdir / "tracks_2d.npy", tracks_np)

    visualize_2d_3d_side_by_side_video(all_frames, tracks_np, P_dyn, args.mesh, outdir / "3d_motion_side_by_side.mp4", fps=args.fps)

    np.save(outdir / "blobs_rest.npy", P_rest.cpu().numpy())
    np.save(outdir / "blobs_frame0.npy", P_final.cpu().numpy())

if __name__ == "__main__":
    main()