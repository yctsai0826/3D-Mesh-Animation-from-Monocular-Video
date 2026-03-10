import os, sys, argparse, json, threading, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Headless-safe plotting
_HAS_PLT = True
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
except Exception:
    _HAS_PLT = False

_HAS_TRIMESH = True
try:
    import trimesh
except Exception:
    _HAS_TRIMESH = False

# Dash dependencies
try:
    import dash
    from dash import dcc, html, Input, Output, State, callback_context
    import plotly.graph_objects as go
    _HAS_DASH = True
except ImportError:
    _HAS_DASH = False

CUR = os.path.dirname(__file__)
if CUR not in sys.path: sys.path.append(CUR)

from utils.graph import build_knn_graph
from models.egnn import SpatialEGNN
from models.tcn import TemporalTCN
from models.head import ResidualHead
from utils.tools import load_ckpt

shutdown_event = threading.Event()

def _permute_for_predict_axis(p_xyz, predict_axis):
    perm = {'x': [0, 1, 2], 'y': [1, 2, 0], 'z': [2, 0, 1]}[predict_axis]
    inv = [0, 0, 0]
    for i, p in enumerate(perm): inv[p] = i
    return p_xyz[..., perm], perm, inv

def load_mesh_glb(mesh_path):
    if not _HAS_TRIMESH or not mesh_path.lower().endswith('.glb'): 
        return None
    try:
        mesh = trimesh.load(mesh_path, force='mesh', process=False)
        if isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) == 0: return None
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        return mesh
    except:
        return None

def extract_edges_from_glb(glb_path):
    if not _HAS_TRIMESH or not glb_path.lower().endswith('.glb'):
        return None
        
    try:
        scene = trimesh.load(glb_path, force='scene')
        node_to_idx = {}
        idx = 0
        
        for node_name in scene.graph.nodes:
            if node_name not in scene.geometry:
                transform, _ = scene.graph.get(node_name)
                pos = transform[:3, 3]
                if np.linalg.norm(pos) > 1e-5:
                    node_to_idx[node_name] = idx
                    idx += 1
                    
        edges = []
        for edge in scene.graph.to_edgelist():
            parent_node, child_node = edge[0], edge[1]
            if parent_node in node_to_idx and child_node in node_to_idx:
                edges.append([node_to_idx[parent_node], node_to_idx[child_node]])
                
        return edges
    except Exception as e:
        print(f"[Topo] 解析 GLB 骨架連線時發生錯誤: {e}")
        return None

def launch_dash_thread(blobs, mesh_path, save_json_path, host='0.0.0.0', port=8050):
    if not _HAS_DASH:
        print("[Error] Dash is not installed. Skipping manual labeling. (pip install dash plotly pandas)")
        shutdown_event.set()
        return

    mesh_x, mesh_y, mesh_z = [], [], []
    mesh = load_mesh_glb(mesh_path)
    if mesh:
        sample_pts, _ = trimesh.sample.sample_surface(mesh, 3000)
        mesh_x, mesh_y, mesh_z = sample_pts[:, 0], sample_pts[:, 1], sample_pts[:, 2]

    app = dash.Dash(__name__, update_title=None)
    
    app.layout = html.Div([
        html.H3("Manual Skeleton Connection Tool", style={'textAlign': 'center'}),
        html.Div([
            html.Div([
                html.H5("操作說明:"),
                html.Ul([
                    html.Li("點擊一個藍色的點來選取它 (變成紅色)。"),
                    html.Li("點擊另一個藍色的點將它們連線。"),
                    html.Li("如果不小心連錯，點擊 'Undo Last Edge' 復原。"),
                    html.Li("完成所有連線後，點擊 'Save & Continue' 繼續程式。"),
                ]),
                html.Hr(),
                html.Button("Undo Last Edge", id="btn-undo", n_clicks=0, style={'marginRight': '10px', 'padding': '10px'}),
                html.Button("Save & Continue", id="btn-save", n_clicks=0, style={'backgroundColor': '#4CAF50', 'color': 'white', 'padding': '10px'}),
                html.Div(id="status-msg", style={'marginTop': '10px', 'fontWeight': 'bold', 'color': 'blue'}),
                html.Pre(id="edge-data-display", style={'marginTop': '10px', 'maxHeight': '300px', 'overflowY': 'scroll', 'border': '1px solid #ccc'})
            ], style={'width': '20%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),
            html.Div([dcc.Graph(id="3d-scatter", style={'height': '85vh'})], style={'width': '78%', 'display': 'inline-block'})
        ]),
        dcc.Store(id='store-edges', data=[]),
        dcc.Store(id='store-selected', data=None),
    ])

    @app.callback(
        [Output("3d-scatter", "figure"), Output("store-edges", "data"), Output("store-selected", "data"),
         Output("edge-data-display", "children"), Output("status-msg", "children")],
        [Input("3d-scatter", "clickData"), Input("btn-undo", "n_clicks"), Input("btn-save", "n_clicks")],
        [State("store-edges", "data"), State("store-selected", "data")]
    )
    def update_graph(clickData, undo_n, save_n, current_edges, selected_idx):
        ctx = callback_context
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else ""
        msg = "準備就緒，請選擇一個點。"
        if current_edges is None: current_edges = []

        if trigger_id == "btn-save":
            os.makedirs(os.path.dirname(os.path.abspath(save_json_path)), exist_ok=True)
            with open(save_json_path, 'w') as f:
                json.dump(current_edges, f)
            shutdown_event.set()
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, "已儲存！您可以關閉此網頁，程式將在終端機繼續執行。"

        if trigger_id == "btn-undo" and len(current_edges) > 0:
            current_edges.pop()
            msg, selected_idx = "已移除最後一條連線。", None 

        elif trigger_id == "3d-scatter" and clickData:
            pt_idx = clickData['points'][0]['pointNumber']
            curve_idx = clickData['points'][0]['curveNumber']
            
            if curve_idx == 1: # Trace 1 是 Blobs
                if selected_idx is None:
                    selected_idx = pt_idx
                    msg = f"已選擇點 {pt_idx}。現在點擊另一個點以連線。"
                else:
                    if selected_idx != pt_idx:
                        edge = sorted([selected_idx, pt_idx])
                        if edge not in current_edges:
                            current_edges.append(edge)
                            msg = f"已連接 {edge[0]} - {edge[1]}。"
                        else:
                            msg = "這條連線已經存在。"
                    selected_idx = None

        fig = go.Figure()
        
        fig.add_trace(go.Scatter3d(x=mesh_x, y=mesh_y, z=mesh_z, mode='markers', 
                                   marker=dict(size=1, color='lightgray', opacity=0.3), name='Mesh Ref', hoverinfo='none'))

        colors = ['red' if i == selected_idx else 'blue' for i in range(len(blobs))]
        sizes = [10 if i == selected_idx else 5 for i in range(len(blobs))]
        fig.add_trace(go.Scatter3d(x=blobs[:,0], y=blobs[:,1], z=blobs[:,2], mode='markers+text',
                                   marker=dict(size=sizes, color=colors), text=[str(i) for i in range(len(blobs))],
                                   textposition="top center", name='Blobs'))

        xe, ye, ze = [], [], []
        for e in current_edges:
            p1, p2 = blobs[e[0]], blobs[e[1]]
            xe.extend([p1[0], p2[0], None])
            ye.extend([p1[1], p2[1], None])
            ze.extend([p1[2], p2[2], None])
        
        fig.add_trace(go.Scatter3d(x=xe, y=ye, z=ze, mode='lines', line=dict(color='green', width=4), name='Skeleton', hoverinfo='none'))
        fig.update_layout(scene=dict(aspectmode='data'), margin=dict(l=0, r=0, b=0, t=0), uirevision='constant')

        return fig, current_edges, selected_idx, "\n".join([f"{e[0]} - {e[1]}" for e in current_edges]), msg

    print(f"\n[Dash] 手 বিষ্ণ動拓樸網頁伺服器已啟動: http://{host}:{port}")
    app.run(host=host, port=port, debug=False, use_reloader=False)


def optimize_rigidity(P_pred, edges, predict_axis='z', lambda_rigid=5.0, lr=0.01, steps=100, device='cuda'):
    if edges is None or len(edges) == 0:
        return P_pred

    print(f"[Optim] 開始剛性物理最佳化 (steps={steps}, weight={lambda_rigid}, 動態軸={predict_axis})...")
    P_target = P_pred.clone().detach().to(device)
    P_opt = P_pred.clone().detach().to(device).requires_grad_(True)
    
    edge_idx_1 = torch.tensor(edges[:, 0], device=device).long()
    edge_idx_2 = torch.tensor(edges[:, 1], device=device).long()
    
    axis_idx = {'x': 0, 'y': 1, 'z': 2}[predict_axis]
    known_axes = [i for i in range(3) if i != axis_idx]
    
    rest_lens = torch.norm(P_target[0, edge_idx_1] - P_target[0, edge_idx_2], dim=-1) # [E]
    
    optimizer = optim.Adam([P_opt], lr=lr)

    for i in range(steps):
        optimizer.zero_grad()
        
        loss_data = torch.mean((P_opt[..., axis_idx] - P_target[..., axis_idx]) ** 2)
        
        curr_lens = torch.norm(P_opt[:, edge_idx_1] - P_opt[:, edge_idx_2], dim=-1) # [T, E]
        
        loss_rigid = torch.mean((curr_lens - rest_lens.unsqueeze(0)) ** 2)
        
        loss = loss_data + lambda_rigid * loss_rigid
        
        loss.backward()

        if P_opt.grad is not None:
            P_opt.grad[..., known_axes] = 0.0
            
        optimizer.step()
        
        if (i+1) % 50 == 0:
            print(f"  Step {i+1}/{steps} | Total Loss: {loss.item():.5f} (Rigid Loss: {loss_rigid.item():.5f})")
            
    return P_opt.detach()

def visualize_xyz_motion_video_skeleton(P_seq, edges, save_path, fps=30, view='topdown', mesh_path=None, point_size=5):
    if not _HAS_PLT: return

    P = P_seq.detach().cpu().numpy()
    T, N, _ = P.shape
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    mins, maxs = P.reshape(-1, 3).min(axis=0), P.reshape(-1, 3).max(axis=0)
    V, F = None, None
    mesh = load_mesh_glb(mesh_path)
    if mesh:
        V, F = np.asarray(mesh.vertices), np.asarray(mesh.faces, dtype=np.int32)
        mins = np.minimum(mins, V.min(axis=0))
        maxs = np.maximum(maxs, V.max(axis=0))

    center = (mins + maxs) / 2
    half = max(maxs - mins) * 0.55 or 1.0

    fig = plt.figure(figsize=(16, 7.5))
    axL = fig.add_subplot(1, 2, 1, projection='3d')
    axR = fig.add_subplot(1, 2, 2, projection='3d')

    for ax in (axL, axR):
        ax.set_xlim(center[0]-half, center[0]+half); ax.set_ylim(center[1]-half, center[1]+half); ax.set_zlim(center[2]-half, center[2]+half)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        if V is not None:
            ax.add_collection3d(Poly3DCollection(V[F], facecolor=(0.8, 0.8, 0.8, 0.05), edgecolor='none'))

    axL.set_title("Result (Left View)")
    axR.set_title("Result (Side View)")

    P0 = P[0]
    sL = axL.scatter(P0[:,0], P0[:,1], P0[:,2], s=point_size, c='C1')
    sR = axR.scatter(P0[:,0], P0[:,1], P0[:,2], s=point_size, c='C1')

    lines_L, lines_R = None, None
    if edges is not None and len(edges) > 0:
        segs = P0[edges]
        lines_L = Line3DCollection(segs, colors='blue', linewidths=0.5, alpha=0.4); axL.add_collection3d(lines_L)
        lines_R = Line3DCollection(segs, colors='blue', linewidths=0.5, alpha=0.4); axR.add_collection3d(lines_R)

    axL.view_init(elev=90, azim=-90) if view == 'topdown' else axL.view_init(elev=20, azim=45)
    axR.view_init(elev=0, azim=0)

    try:
        img_path = save_path.replace('.mp4', '_frame0.png')
        plt.savefig(img_path, dpi=150)
        print(f"[VIZ] Frame 0 saved: {img_path}")
    except Exception as e: pass

    def update(t):
        sL._offsets3d, sR._offsets3d = tuple(P[t].T), tuple(P[t].T)
        if edges is not None and len(edges) > 0:
            lines_L.set_segments(P[t][edges]); lines_R.set_segments(P[t][edges])
        if view == 'orbit': axL.view_init(elev=20, azim=(45 + t * 0.8) % 360)
        return sL, sR, lines_L, lines_R

    try:
        print(f"[VIZ] Rendering video ({T} frames)...")
        animation.FuncAnimation(fig, update, frames=T, interval=1000/fps).save(str(save_path), writer='ffmpeg', fps=fps)
        print(f"[VIZ] Video saved: {save_path}")
    except Exception as e:
        print("[VIZ] Video failed:", e)
    finally:
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    # Explicit File Paths
    ap.add_argument('--input_motion', type=str, required=True, help='Path to input 3d_motion_blobs.pth')
    
    ap.add_argument('--outdir', type=str, default='./output/2_gen3dblobs', help='Output directory for generated files')
    
    ap.add_argument('--edges_json', type=str, default=None, help='Path to save/load skeleton edges (If None, extract from GLB)')
    ap.add_argument('--mesh', type=str, required=True, help='Path to reference .glb mesh')
    ap.add_argument('--ckpt', type=str, required=True, help='Model ckpt path')
    
    # Model Args
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--predict_axis', type=str, default='z', choices=['x','y','z'])
    
    # Rigidity args
    ap.add_argument('--apply_rigidity', type=int, default=1, help='Apply rigid constraint')
    ap.add_argument('--rigidity_weight', type=float, default=10.0, help='Weight for rigidity loss')
    ap.add_argument('--optim_steps', type=int, default=100)
    
    # Manual Topo Args
    ap.add_argument('--dash_port', type=int, default=8050, help='Port for manual tool')
    ap.add_argument('--skip_manual_if_exists', type=int, default=1, help='Skip dash if json exists (Default: 1)')

    # Video args
    ap.add_argument('--make_video', type=int, default=1)
    ap.add_argument('--video_fps', type=int, default=30)
    ap.add_argument('--view', type=str, default='topdown')
    ap.add_argument('--point_size', type=int, default=5)

    args = ap.parse_args()
    dev = torch.device(args.device)
    
    os.makedirs(args.outdir, exist_ok=True)
    out_motion_path = os.path.join(args.outdir, '3d_motion_skeleton.pth')
    out_video_path = os.path.join(args.outdir, 'pred_motion_skeleton.mp4')

    # 1) Load Input Motion
    if not os.path.isfile(args.input_motion): 
        raise FileNotFoundError(f"Missing input: {args.input_motion}")
    P_in = torch.load(args.input_motion, map_location='cpu') # [T, N, 3]
    T, N, _ = P_in.shape
    print(f"[INFO] Loaded Input Motion: {P_in.shape}")

    # 2) Predict Missing Axis
    print(f"[INFO] Predicting missing axis '{args.predict_axis}' with model...")
    P_rot, perm, inv = _permute_for_predict_axis(P_in, args.predict_axis)
    ck = load_ckpt(args.ckpt, map_location=dev)
    
    ckpt_dict = ck.get('model_state_dict', ck.get('state_dict', ck))
    ck_args = ck.get('args', {}) if isinstance(ck, dict) else {}
    
    spatial = SpatialEGNN(in_dim=5, hidden_dim=ck_args.get('hidden',64), layers=ck_args.get('egnn_layers',2), update_coords=False).to(dev)
    temporal = TemporalTCN(in_dim=ck_args.get('hidden',64), hidden_dim=ck_args.get('tcn_hidden',128), out_dim=ck_args.get('tcn_out',128), n_layers=ck_args.get('tcn_layers',6)).to(dev)
    head = ResidualHead(in_dim=ck_args.get('tcn_out',128), hidden=ck_args.get('head_hidden',128)).to(dev)
    
    if 'spatial' in ckpt_dict:
        spatial.load_state_dict(ckpt_dict['spatial'], strict=False)
        temporal.load_state_dict(ckpt_dict['temporal'], strict=False)
        head.load_state_dict(ckpt_dict['head'], strict=False)
    else:
        spatial_sd = {k.replace('spatial.', ''): v for k, v in ckpt_dict.items() if k.startswith('spatial.')}
        temporal_sd = {k.replace('temporal.', ''): v for k, v in ckpt_dict.items() if k.startswith('temporal.')}
        head_sd = {k.replace('head.', ''): v for k, v in ckpt_dict.items() if k.startswith('head.')}
        spatial.load_state_dict(spatial_sd if spatial_sd else ckpt_dict, strict=False)
        temporal.load_state_dict(temporal_sd if temporal_sd else ckpt_dict, strict=False)
        head.load_state_dict(head_sd if head_sd else ckpt_dict, strict=False)

    spatial.eval(); temporal.eval(); head.eval()

    Xrot = P_rot.unsqueeze(0).to(dev)
    x0 = Xrot[:, 0:1, :, 0:1].repeat(1, T, 1, 1)
    yz = Xrot[..., 1:3]
    
    seq_mask = torch.ones((1, T, 1, 1), dtype=torch.bool, device=dev)
    mask_flag, pad_flag = seq_mask.expand(-1,-1,N,1).float(), (~seq_mask).expand(-1,-1,N,1).float()
    feats = torch.cat([yz, x0, mask_flag, pad_flag], dim=-1)
    
    with torch.no_grad():
        edge_index = build_knn_graph(Xrot[0,0], k=ck_args.get('k', 6))
        xhat = head(temporal(spatial(feats, Xrot, edge_index)), x0)

    Xrot_pred = Xrot.clone()
    Xrot_pred[..., 0:1] = xhat
    P_out = Xrot_pred[..., inv].squeeze(0).cpu()

    # 3) Manual Topology Labeling (Dash) & Rigidity Optimization
    edges_arr = None
    
    edges_json_path = args.edges_json if args.edges_json else os.path.join(args.outdir, 'skeleton_edges.json')

    if args.apply_rigidity:
        blobs_rest = P_out[0].numpy()
        need_labeling = False
        
        if args.edges_json is None:
            extracted_edges = extract_edges_from_glb(args.mesh)
            if extracted_edges is not None and len(extracted_edges) > 0:
                edges_arr = np.array(extracted_edges)
                print(f"[Topo] 從 RigAnything GLB 自動提取了 {len(edges_arr)} 條骨架連線！跳過網頁手動標註。")
            else:
                print(f"[Topo] 自動提取骨架失敗，或這不是一個帶有骨架的 GLB。將啟動手動標註工具...")
                need_labeling = True
        else:
            if args.skip_manual_if_exists and os.path.isfile(edges_json_path):
                print(f"[Topo] 發現已存在的骨架存檔: {edges_json_path}。跳過網頁手動標註。")
                with open(edges_json_path, 'r') as f:
                    edges_list = json.load(f)
                    if len(edges_list) > 0:
                        edges_arr = np.array(edges_list)
            else:
                need_labeling = True
            
        if need_labeling:
            shutdown_event.clear()
            t = threading.Thread(target=launch_dash_thread, args=(blobs_rest, args.mesh, edges_json_path, '0.0.0.0', args.dash_port))
            t.daemon = True 
            t.start()
            
            print("\n" + "="*60)
            print("="*60 + "\n")
            
            try:
                while not shutdown_event.is_set(): time.sleep(1)
            except KeyboardInterrupt:
                sys.exit(0)

            if os.path.isfile(edges_json_path):
                with open(edges_json_path, 'r') as f:
                    edges_list = json.load(f)
                if len(edges_list) > 0:
                    edges_arr = np.array(edges_list)

        if edges_arr is not None and len(edges_arr) > 0:
            P_out = optimize_rigidity(
                P_pred=P_out.to(dev), 
                edges=edges_arr, 
                predict_axis=args.predict_axis, 
                lambda_rigid=args.rigidity_weight, 
                lr=args.optim_lr if hasattr(args, 'optim_lr') else 0.01, 
                steps=args.optim_steps, 
                device=dev
            ).cpu()
        else:
            print("[Topo] ignore Rigidity Optimization。")

    # 4) Save & Viz
    torch.save(P_out, out_motion_path)
    print(f"[SAVE] Saved motion trajectory to {out_motion_path}")
    
    out_motion_npy_path = os.path.join(args.outdir, '3d_motion_skeleton.npy')
    np.save(out_motion_npy_path, P_out.detach().cpu().numpy())
    print(f"[SAVE] Saved motion trajectory (Numpy format) to {out_motion_npy_path}")
    if args.make_video:
        visualize_xyz_motion_video_skeleton(P_out, edges_arr, save_path=out_video_path, fps=args.video_fps, view=args.view, mesh_path=args.mesh, point_size=args.point_size)

if __name__ == '__main__':
    main()