import os
import sys
import argparse
import json
import threading
import time
import numpy as np
import trimesh
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
    import dash
    from dash import dcc, html, Input, Output, State, callback_context
    import plotly.graph_objects as go
    _HAS_DASH = True
except ImportError:
    _HAS_DASH = False

try:
    import bpy
    from mathutils import Vector
except ImportError:
    sys.exit(1)

shutdown_event = threading.Event()

def log(msg):
    print(f"[Pipeline] {msg}", flush=True)

def yup_to_zup(pts):
    pts = np.asarray(pts).copy()
    res = pts.copy()
    res[..., 1] = -pts[..., 2]
    res[..., 2] = pts[..., 1]
    return res

def smooth_motion_data(joints, window_size=5):
    if window_size < 3:
        return joints
    
    log(f"Smoothing motion data with window size: {window_size}...")
    pad_width = window_size // 2
    smoothed = np.copy(joints)
    
    for k in range(joints.shape[1]):
        for d in range(joints.shape[2]):
            series = joints[:, k, d]
            padded = np.pad(series, (pad_width, pad_width), mode='edge')
            smoothed[:, k, d] = np.convolve(padded, np.ones(window_size)/window_size, mode='valid')
            
    return smoothed

def ensure_object_mode():
    if bpy.context.active_object and bpy.context.active_object.mode != 'OBJECT':
        try: bpy.ops.object.mode_set(mode='OBJECT')
        except: pass

def make_active(obj):
    ensure_object_mode()
    bpy.ops.object.select_all(action='DESELECT')
    real_obj = bpy.data.objects.get(obj.name, obj)
    real_obj.select_set(True)
    bpy.context.view_layer.objects.active = real_obj
    return real_obj

def select_many(objs, active=None, deselect_first=True):
    ensure_object_mode()
    if deselect_first: bpy.ops.object.select_all(action='DESELECT')
    for o in objs:
        if o is None: continue
        real = bpy.data.objects.get(o.name, o)
        real.select_set(True)
    if active is None and len(objs) > 0: active = objs[0]
    if active is not None:
        bpy.context.view_layer.objects.active = bpy.data.objects.get(active.name, active)

def clean_mesh_geometry(mesh_obj):
    ensure_object_mode()
    make_active(mesh_obj)
    try:
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.normals_make_consistent(inside=False)
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')

def extract_edges_from_glb(glb_path):
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
            if edge[0] in node_to_idx and edge[1] in node_to_idx:
                edges.append([node_to_idx[edge[0]], node_to_idx[edge[1]]])
        return edges
    except Exception as e:
        log(f"[Warning] Failed to extract edges from GLB: {e}")
        return None


def launch_alignment_dash(mesh_path, joints, save_json_path, host='0.0.0.0', port=8050):
    if not _HAS_DASH:
        log("[Error] Lack of Dash library. Please install it with 'pip install dash' and run this script in a compatible environment.")
        shutdown_event.set()
        return

    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    sample_pts, _ = trimesh.sample.sample_surface(mesh, 3000)
    sample_pts = yup_to_zup(sample_pts)
    joints_0 = joints[0]

    app = dash.Dash(__name__, update_title=None)

    app.layout = html.Div([
        html.H3("Mesh & Skeleton 對齊工具", style={'textAlign': 'center', 'fontFamily': 'sans-serif'}),
        html.Div([
            html.Div([
                html.H5("調整參數讓灰色 Mesh 包覆紅色 Skeleton:"),
                html.Label("Scale (縮放):"), dcc.Input(id='in-scale', type='number', value=1.0, step=0.01, style={'width': '100%', 'marginBottom': '10px'}),
                html.Label("Translate X (左右):"), dcc.Input(id='in-tx', type='number', value=0.0, step=0.01, style={'width': '100%', 'marginBottom': '10px'}),
                html.Label("Translate Y (前後):"), dcc.Input(id='in-ty', type='number', value=0.0, step=0.01, style={'width': '100%', 'marginBottom': '10px'}),
                html.Label("Translate Z (上下):"), dcc.Input(id='in-tz', type='number', value=0.0, step=0.01, style={'width': '100%', 'marginBottom': '20px'}),
                html.Hr(),
                html.Button("儲存並繼續 (Save & Continue)", id="btn-save", n_clicks=0, style={'backgroundColor': '#4CAF50', 'color': 'white', 'padding': '12px', 'width': '100%', 'fontSize': '16px'}),
                html.Div(id='status-msg', style={'marginTop': '15px', 'color': 'blue', 'fontWeight': 'bold'})
            ], style={'width': '20%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px', 'boxSizing': 'border-box'}),
            html.Div([dcc.Graph(id='3d-plot', style={'height': '85vh'})], style={'width': '78%', 'display': 'inline-block'})
        ])
    ])

    @app.callback(
        [Output('3d-plot', 'figure'), Output('status-msg', 'children')],
        [Input('in-scale', 'value'), Input('in-tx', 'value'), Input('in-ty', 'value'), Input('in-tz', 'value'), Input('btn-save', 'n_clicks')]
    )
    def update_graph(scale, tx, ty, tz, n_clicks):
        ctx = callback_context
        trigger = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else ""
        scale = float(scale) if scale is not None else 1.0
        tx, ty, tz = float(tx or 0), float(ty or 0), float(tz or 0)

        msg = ""
        if trigger == "btn-save":
            with open(save_json_path, 'w') as f: json.dump({"scale": scale, "tx": tx, "ty": ty, "tz": tz}, f)
            shutdown_event.set()
            msg = "Saved alignment parameters. You can now close this page and return to the pipeline."

        trans_pts = sample_pts * scale + np.array([tx, ty, tz], dtype=np.float32)
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=trans_pts[:, 0], y=trans_pts[:, 1], z=trans_pts[:, 2], mode='markers', marker=dict(size=2, color='gray', opacity=0.3), name='Mesh'))
        fig.add_trace(go.Scatter3d(x=joints_0[:, 0], y=joints_0[:, 1], z=joints_0[:, 2], mode='markers', marker=dict(size=6, color='red'), name='Skeleton'))
        fig.update_layout(scene=dict(aspectmode='data'), margin=dict(l=0, r=0, b=0, t=0))
        return fig, msg

    log(f"Open web browser: http://{host}:{port}")
    app.run(host=host, port=port, debug=False, use_reloader=False)

def visualize_alignment(mesh_path, joints, align_params, out_png):
    mesh = trimesh.load(mesh_path, force='mesh')
    if isinstance(mesh, trimesh.Scene): mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    V = yup_to_zup(np.asarray(mesh.vertices))
    s, tx, ty, tz = align_params['scale'], align_params['tx'], align_params['ty'], align_params['tz']
    V_aligned = V * s + np.array([tx, ty, tz])

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])
    ax.add_collection3d(Poly3DCollection(V_aligned[np.asarray(mesh.faces)], facecolor='gray', alpha=0.15, edgecolor='none'))
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='red', s=40)

    mins, maxs = np.minimum(V_aligned.min(0), joints.min(0)), np.maximum(V_aligned.max(0), joints.max(0))
    c, span = (mins + maxs) / 2, (maxs - mins).max() / 2 * 1.1
    ax.set_xlim(c[0]-span, c[0]+span); ax.set_ylim(c[1]-span, c[1]+span); ax.set_zlim(c[2]-span, c[2]+span)
    ax.view_init(elev=20, azim=-45)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def build_parents_from_edges(num_joints, edges):
    parents = np.full(num_joints, -1, dtype=np.int64)
    if not edges: return parents
    adj = {i: [] for i in range(num_joints)}
    for u, v in edges:
        adj[u].append(v); adj[v].append(u)
    visited, queue = set([0]), [0]
    while queue:
        curr = queue.pop(0)
        for neighbor in adj[curr]:
            if neighbor not in visited:
                visited.add(neighbor); parents[neighbor] = curr; queue.append(neighbor)
    return parents

def build_armature(joints0_100, parents, arm_name="Armature"):
    bpy.ops.object.add(type="ARMATURE", enter_editmode=True)
    arm_obj = bpy.context.object
    arm_obj.name = arm_name
    arm = arm_obj.data
    for b in list(arm.edit_bones): arm.edit_bones.remove(b)

    K = joints0_100.shape[0]; bones = [None] * K

    for i in range(K):
        p = int(parents[i])
        b = arm.edit_bones.new(f"joint_{i}")
        if p < 0 or p == i:
            head = Vector(joints0_100[i].tolist())
            tail = head + Vector((0.0, 0.0, 5.0))
        else:
            head = Vector(joints0_100[p].tolist())
            tail = Vector(joints0_100[i].tolist())
            if (tail - head).length < 1e-8:
                tail = head + Vector((0.0, 0.0, 5.0))

        b.head, b.tail = head, tail
        b.head_radius = 2.0; b.tail_radius = 2.0; b.envelope_distance = 1.0
        bones[i] = b

    for i in range(K):
        p = int(parents[i])
        if p >= 0 and p != i and bones[i] and bones[p]:
            bones[i].parent = bones[p]; bones[i].use_connect = False
            try: bones[i].inherit_scale = 'NONE'
            except: pass

    bpy.ops.object.mode_set(mode="OBJECT")
    return arm_obj

def bind_with_voxel_proxy(mesh_obj, arm_obj):
    log("Binding Mesh to Armature using Voxel Proxy + DataTransfer (robust)...")
    ensure_object_mode()

    make_active(mesh_obj)
    bpy.ops.object.duplicate()
    proxy_obj = bpy.context.active_object
    proxy_obj.name = "Mesh_Proxy"

    max_dim = max(proxy_obj.dimensions)
    voxel_size = max_dim / 200.0  
    remesh_mod = proxy_obj.modifiers.new(name="VoxelRemesh", type='REMESH')
    remesh_mod.mode = 'VOXEL'; remesh_mod.voxel_size = voxel_size; remesh_mod.adaptivity = 0.0
    make_active(proxy_obj)
    try: bpy.ops.object.modifier_apply(modifier=remesh_mod.name)
    except: log("[Warn] Failed to apply voxel remesh modifier; continue anyway.")

    ensure_object_mode()
    select_many([proxy_obj, arm_obj], active=arm_obj, deselect_first=True)

    ok_auto = True
    try: bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    except Exception: ok_auto = False

    proxy_has_weights = (len(proxy_obj.vertex_groups) > 0) and any(len(v.groups) > 0 for v in proxy_obj.data.vertices)

    if (not ok_auto) or (not proxy_has_weights):
        log("Proxy Heat Weight failed. Fallback to ARMATURE_ENVELOPE...")
        ensure_object_mode()
        try: make_active(proxy_obj); bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
        except: pass

        ensure_object_mode()
        select_many([proxy_obj, arm_obj], active=arm_obj, deselect_first=True)
        try: bpy.ops.object.parent_set(type="ARMATURE_ENVELOPE")
        except: pass

    for g in proxy_obj.vertex_groups:
        if g.name not in mesh_obj.vertex_groups: mesh_obj.vertex_groups.new(name=g.name)

    ensure_object_mode()
    select_many([mesh_obj, arm_obj], active=arm_obj, deselect_first=True)
    try: bpy.ops.object.parent_set(type="ARMATURE")  
    except: pass
    
    arm_mod = None
    for m in mesh_obj.modifiers:
        if m.type == 'ARMATURE' and m.object == arm_obj: arm_mod = m
    if not arm_mod:
        arm_mod = mesh_obj.modifiers.new(name="Armature", type='ARMATURE')
        arm_mod.object = arm_obj
    try: arm_mod.use_deform_preserve_volume = True
    except: pass

    ensure_object_mode()
    make_active(mesh_obj)
    dt_mod = mesh_obj.modifiers.new(name="WeightTransfer", type='DATA_TRANSFER')
    dt_mod.object = proxy_obj; dt_mod.use_vert_data = True
    dt_mod.data_types_verts = {'VGROUP_WEIGHTS'}; dt_mod.vert_mapping = 'NEAREST'  

    try:
        bpy.ops.object.modifier_apply(modifier=dt_mod.name)
        log("Weight transfer applied to mesh.")
    except Exception as e:
        log(f"Failed to apply DataTransfer modifier: {e}")

    try: bpy.data.objects.remove(proxy_obj, do_unlink=True)
    except: pass


def create_animated_empties(joints, names):
    T, K, _ = joints.shape
    empties = []
    collection = bpy.data.collections.new("Joint_Targets")
    bpy.context.scene.collection.children.link(collection)

    for i in range(K):
        o = bpy.data.objects.new(names[i] + "_target", None)
        o.empty_display_type = 'PLAIN_AXES'; o.empty_display_size = 5.0
        collection.objects.link(o)
        empties.append(o)

    bpy.context.scene.frame_start, bpy.context.scene.frame_end = 0, T - 1

    for t in range(T):
        for i in range(K):
            empties[i].location = joints[t, i]
            empties[i].keyframe_insert(data_path="location", frame=t)
    return empties

def add_constraints_to_armature(arm_obj, parents, empties, joint_names):
    K = len(parents)
    bpy.context.scene.frame_set(0)
    bpy.context.view_layer.update()

    make_active(arm_obj)
    bpy.ops.object.mode_set(mode='POSE')

    for i in range(K):
        p = parents[i]
        bone_name = joint_names[i]
        if bone_name not in arm_obj.pose.bones: continue
        p_bone = arm_obj.pose.bones[bone_name]

        for c in list(p_bone.constraints): p_bone.constraints.remove(c)

        if p == -1 or p == i:
            c_loc = p_bone.constraints.new('COPY_LOCATION')
            c_loc.target = empties[i]
        else:
            c_loc = p_bone.constraints.new('COPY_LOCATION')
            c_loc.target = empties[p]
            c_stretch = p_bone.constraints.new('STRETCH_TO')
            c_stretch.target = empties[i]
            c_stretch.volume = 'NO_VOLUME'

    bpy.ops.object.mode_set(mode='OBJECT')

def bake_animation_and_cleanup(arm_obj, frame_end):
    log("Baking animation to keyframes...")
    make_active(arm_obj)
    bpy.ops.object.mode_set(mode='POSE')
    bpy.ops.pose.select_all(action='SELECT')
    
    bpy.ops.nla.bake(
        frame_start=0, frame_end=frame_end, only_selected=True,
        visual_keying=True, clear_constraints=True, use_current_action=True, bake_types={'POSE'}
    )
    bpy.ops.object.mode_set(mode='OBJECT')


def setup_camera_and_render(out_mp4, fps, arm_obj, empties, joints_all):
    log("Setting up Camera and Lights for Rendering...")
    scene = bpy.context.scene

    scene.render.resolution_x, scene.render.resolution_y = 1024, 1024
    
    world = scene.world
    if not world: world = bpy.data.worlds.new("World"); scene.world = world
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get('Background')
    if bg_node: bg_node.inputs[0].default_value = (0.25, 0.25, 0.25, 1.0)

    mesh_objs = [o for o in bpy.data.objects if o.type == 'MESH']
    if not mesh_objs: return
    
    coords = joints_all.reshape(-1, 3)
    xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]
    cx, cy, cz = sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs)
    max_span = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs))
    safe_span = max_span * 2.5

    bpy.ops.object.camera_add()
    cam = bpy.context.object; scene.camera = cam
    cam.data.clip_start = 0.1; cam.data.clip_end = 1000000.0
    cam.data.angle = math.radians(45) 
    
    distance = (safe_span / 2.0) / math.tan(cam.data.angle / 2.0)
    cam.location = (cx, cy - distance, cz + safe_span * 0.1)

    bpy.ops.object.empty_add(location=(cx, cy, cz)); target = bpy.context.object
    c_track = cam.constraints.new('TRACK_TO'); c_track.target = target
    c_track.track_axis = 'TRACK_NEGATIVE_Z'; c_track.up_axis = 'UP_Y'

    bpy.ops.object.light_add(type='SUN', location=(cx - safe_span, cy - safe_span, cz + safe_span)); bpy.context.object.data.energy = 4.0
    bpy.ops.object.light_add(type='SUN', location=(cx + safe_span, cy - safe_span, cz + safe_span*0.5)); bpy.context.object.data.energy = 1.5
    bpy.ops.object.light_add(type='SUN', location=(cx, cy + safe_span, cz + safe_span)); bpy.context.object.data.energy = 2.5

    try: scene.render.engine = 'BLENDER_EEVEE_NEXT'
    except TypeError: scene.render.engine = 'BLENDER_EEVEE'

    scene.render.fps = fps; scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = 'MPEG4'; scene.render.ffmpeg.codec = 'H264'

    out_debug = out_mp4.replace(".mp4", "_debug_armature.mp4")
    
    hidden_states = {}
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            hidden_states[obj] = obj.hide_render
            obj.hide_render = True
            
    debug_meshes = []
    mat_target = bpy.data.materials.new(name="TargetMat"); mat_target.use_nodes = True
    if "Principled BSDF" in mat_target.node_tree.nodes:
        mat_target.node_tree.nodes["Principled BSDF"].inputs['Base Color'].default_value = (0.1, 0.3, 1.0, 1.0) 
    mat_bone = bpy.data.materials.new(name="BoneMat"); mat_bone.use_nodes = True
    if "Principled BSDF" in mat_bone.node_tree.nodes:
        mat_bone.node_tree.nodes["Principled BSDF"].inputs['Base Color'].default_value = (1.0, 0.8, 0.1, 1.0) 

    for emp in empties:
        bpy.ops.mesh.primitive_ico_sphere_add(radius=1.5, subdivisions=2)
        sph = bpy.context.object; sph.data.materials.append(mat_target)
        c = sph.constraints.new('COPY_LOCATION'); c.target = emp
        debug_meshes.append(sph)
        
    for bone in arm_obj.pose.bones:
        bpy.ops.mesh.primitive_ico_sphere_add(radius=2.0, subdivisions=2)
        sph_h = bpy.context.object; sph_h.data.materials.append(mat_bone)
        c = sph_h.constraints.new('COPY_LOCATION')
        c.target = arm_obj; c.subtarget = bone.name; c.head_tail = 0.0
        debug_meshes.append(sph_h)

    # scene.render.filepath = str(out_debug)
    # bpy.ops.render.render(animation=True)
    
    for obj in debug_meshes: bpy.data.objects.remove(obj, do_unlink=True)
    for obj, state in hidden_states.items(): obj.hide_render = state

    scene.render.filepath = str(out_mp4)
    bpy.ops.render.render(animation=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--motion", required=True)
    parser.add_argument("--edges", default=None)
    parser.add_argument("--out_dir", default="./output")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument('--dash_port', type=int, default=8051)
    parser.add_argument("--smooth_window", type=int, default=5)

    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    out_blend = os.path.join(args.out_dir, "output.blend")
    out_mp4 = os.path.join(args.out_dir, "output.mp4")
    out_debug_mp4 = os.path.join(args.out_dir, "debug_armature_motion.mp4")
    out_png = os.path.join(args.out_dir, "alignment_check.png")
    align_json = os.path.join(args.out_dir, "alignment_params.json")

    log("Loading Motion Data...")
    if args.motion.endswith('.npy'): joints_all = np.load(args.motion)
    else:
        import torch
        joints_all = torch.load(args.motion, map_location='cpu').numpy()

    joints_all = yup_to_zup(joints_all)
    joints_all = smooth_motion_data(joints_all, window_size=args.smooth_window)
    
    joints0 = joints_all[0]
    K = joints0.shape[0]

    edges = None
    if args.edges and os.path.isfile(args.edges):
        with open(args.edges, 'r') as f:
            edges = json.load(f)
    else:
        edges = extract_edges_from_glb(args.mesh)
        if edges: log(f"successfully extracted {len(edges)} edges!")
        else: log("[Warning] Failed to extract skeleton edges, will use linear connection.")
            
    parents = build_parents_from_edges(K, edges)

    # --- 2. Dash Alignment Check ---
    if not os.path.exists(align_json):
        shutdown_event.clear()
        t = threading.Thread(target=launch_alignment_dash, args=(args.mesh, joints_all, align_json, '0.0.0.0', args.dash_port))
        t.daemon = True; t.start()
        try:
            while not shutdown_event.is_set(): time.sleep(1)
        except KeyboardInterrupt: sys.exit(0)

    with open(align_json, 'r') as f: align_params = json.load(f)
    visualize_alignment(args.mesh, joints0, align_params, out_png)

    s = float(align_params['scale'])
    tx, ty, tz = float(align_params['tx']), float(align_params['ty']), float(align_params['tz'])
    
    joints_all_100 = joints_all * 100.0
    joints0_100 = joints_all_100[0]

    log("Starting Blender Subsystem...")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=args.mesh)

    for obj in bpy.context.scene.objects:
        obj.rotation_euler = (0, 0, 0); obj.location = (0, 0, 0)

    mesh_objs = [o for o in bpy.data.objects if o.type == "MESH"]
    if not mesh_objs: raise RuntimeError("No MESH objects found in GLB.")

    for arm in [o for o in bpy.data.objects if o.type == 'ARMATURE']:
        bpy.data.objects.remove(arm, do_unlink=True)
        
    mesh_objs_sorted = sorted(mesh_objs, key=lambda o: len(o.data.vertices), reverse=True)
    active_mesh = mesh_objs_sorted[0]
    select_many(mesh_objs_sorted, active=active_mesh, deselect_first=True)

    if len(mesh_objs_sorted) > 1: bpy.ops.object.join()
    mesh_obj = bpy.context.view_layer.objects.active
    
    mesh_obj.vertex_groups.clear()
    for mod in list(mesh_obj.modifiers):
        if mod.type == 'ARMATURE': mesh_obj.modifiers.remove(mod)

    make_active(mesh_obj)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    
    for v in mesh_obj.data.vertices:
        v.co.x = (v.co.x * s + tx) * 100.0
        v.co.y = (v.co.y * s + ty) * 100.0
        v.co.z = (v.co.z * s + tz) * 100.0
    mesh_obj.data.update()
    
    clean_mesh_geometry(mesh_obj)

    arm_obj = build_armature(joints0_100, parents)
    bind_with_voxel_proxy(mesh_obj, arm_obj)

    joint_names = [f"joint_{i}" for i in range(K)]
    empties = create_animated_empties(joints_all_100, joint_names)
    
    add_constraints_to_armature(arm_obj, parents, empties, joint_names)
    
    bake_animation_and_cleanup(arm_obj, len(joints_all_100) - 1)

    setup_camera_and_render(out_mp4, args.fps, arm_obj, empties, joints_all_100)

    for e in empties:
        try: bpy.data.objects.remove(e, do_unlink=True)
        except: pass

    mesh_obj.name = "Character_Mesh"
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D': space.clip_end = 100000.0

    bpy.ops.wm.save_as_mainfile(filepath=out_blend)
    log(f"Pipeline Complete! Check {args.out_dir} for outputs.")

if __name__ == "__main__":
    main()