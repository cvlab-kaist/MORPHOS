"""
Prepare training data for dynamic meshes (animated GLBs) for TRELLIS training.

Two-pass pipeline:
  Pass 1 (Blender): render multi-view RGBA + export per-frame meshes
  Pass 2 (PyTorch): voxelize, extract DINOv2 features, encode SS + SLat latents

Uses TRELLIS's own encoders. Output structure matches TRELLIS conventions.

Usage:
  # Pass 1: Blender rendering
  blender --background --python dataset_toolkits/process_tlsat_dataset.py -- \\
      --pass render --glb_dir /path/to/glbs --output_dir /path/to/output \\
      --rank 0 --world_size 1 -n 10

  # Pass 2: PyTorch encoding
  python dataset_toolkits/process_tlsat_dataset.py \\
      --pass encode --output_dir /path/to/output \\
      --rank 0 --world_size 1 -n 10

  # Or use the launcher: bash dataset_toolkits/run_process_tslat_dataset.sh render 10
"""

import os
import sys
import json
import argparse
import shutil
import math
import zlib
import numpy as np
from pathlib import Path


def log_progress(args, msg, is_error=False):
    print(msg, flush=True)
    if is_error:
        err_path = os.path.join(args.output_dir, 'logs', f'{args.pipeline_pass}_errors.log')
        os.makedirs(os.path.dirname(err_path), exist_ok=True)
        with open(err_path, 'a') as f:
            f.write(msg + '\n')

def update_status(args, msg):
    status_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(status_dir, exist_ok=True)
    with open(os.path.join(status_dir, f'status_{args.pipeline_pass}_rank{args.rank}.txt'), 'w') as f:
        f.write(msg + '\n')


# ============================================================================
# Pass 1: Blender rendering
# ============================================================================

def pass_render(args):
    import bpy

    def clear_scene():
        for obj in bpy.data.objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        for block in bpy.data.meshes:
            bpy.data.meshes.remove(block)
        for block in bpy.data.materials:
            bpy.data.materials.remove(block)
        for block in bpy.data.textures:
            bpy.data.textures.remove(block)
        for block in bpy.data.images:
            bpy.data.images.remove(block)
        for block in bpy.data.actions:
            bpy.data.actions.remove(block)
        for block in bpy.data.cameras:
            bpy.data.cameras.remove(block)
        for block in bpy.data.lights:
            bpy.data.lights.remove(block)

    def remove_junk_objects():
        """Remove Blender-default icosphere helper meshes (42 verts, used as
        light/camera proxies in some GLBs). Only removes MESH-type objects
        whose base name (stripping Blender's .NNN suffix) is 'icosphere'.
        Does NOT remove 'cube' — real character meshes can be named Cube,
        and parent EMPTYs named Cube would orphan the hierarchy if removed."""
        import re
        junk_names = {'icosphere'}
        suffix_re = re.compile(r'\.\d+$')
        for obj in list(bpy.context.scene.objects):
            if obj.type != 'MESH':
                continue
            base = suffix_re.sub('', obj.name.lower())
            if base in junk_names:
                bpy.data.objects.remove(obj, do_unlink=True)
        bpy.context.view_layer.update()

    def setup_renderer(resolution):
        scene = bpy.context.scene
        scene.render.engine = 'CYCLES'
        scene.cycles.device = 'GPU'
        scene.cycles.samples = 128
        scene.cycles.use_denoising = True
        scene.render.resolution_x = resolution
        scene.render.resolution_y = resolution
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGBA'
        scene.render.film_transparent = True
        scene.cycles.filter_type = 'BOX'
        scene.cycles.filter_width = 1
        scene.cycles.diffuse_bounces = 1
        scene.cycles.glossy_bounces = 1
        scene.cycles.transparent_max_bounces = 3
        scene.cycles.transmission_bounces = 3
        prefs = bpy.context.preferences
        prefs.addons['cycles'].preferences.get_devices()
        prefs.addons['cycles'].preferences.compute_device_type = 'CUDA'
        for device in prefs.addons['cycles'].preferences.devices:
            device.use = False
        cuda_devices = [d for d in prefs.addons['cycles'].preferences.devices if d.type == 'CUDA']
        if cuda_devices:
            cuda_devices[args.rank % len(cuda_devices)].use = True

    def setup_lighting():
        bpy.ops.object.select_all(action="DESELECT")
        bpy.ops.object.select_by_type(type="LIGHT")
        bpy.ops.object.delete()
        key = bpy.data.objects.new("Default_Light", bpy.data.lights.new("Default_Light", type="POINT"))
        bpy.context.collection.objects.link(key)
        key.data.energy = 1000
        key.location = (4, 1, 6)
        top = bpy.data.objects.new("Top_Light", bpy.data.lights.new("Top_Light", type="AREA"))
        bpy.context.collection.objects.link(top)
        top.data.energy = 10000
        top.location = (0, 0, 10)
        top.scale = (100, 100, 100)
        bottom = bpy.data.objects.new("Bottom_Light", bpy.data.lights.new("Bottom_Light", type="AREA"))
        bpy.context.collection.objects.link(bottom)
        bottom.data.energy = 1000
        bottom.location = (0, 0, -10)

    def init_camera(track_target=(0, 0, 0)):
        cam = bpy.data.objects.new('Camera', bpy.data.cameras.new('Camera'))
        bpy.context.collection.objects.link(cam)
        bpy.context.scene.camera = cam
        cam.data.sensor_height = cam.data.sensor_width = 32
        constraint = cam.constraints.new(type='TRACK_TO')
        constraint.track_axis = 'TRACK_NEGATIVE_Z'
        constraint.up_axis = 'UP_Y'
        empty = bpy.data.objects.new("Empty", None)
        empty.location = track_target
        bpy.context.scene.collection.objects.link(empty)
        constraint.target = empty
        return cam, empty

    def get_transform_matrix(obj):
        pos, rt, _ = obj.matrix_world.decompose()
        rt = rt.to_matrix()
        matrix = []
        for ii in range(3):
            a = [rt[ii][jj] for jj in range(3)]
            a.append(pos[ii])
            matrix.append(a)
        matrix.append([0, 0, 0, 1])
        return matrix

    def get_animation_frame_range():
        all_frames = set()
        for obj in bpy.data.objects:
            if obj.animation_data and obj.animation_data.action:
                for fc in obj.animation_data.action.fcurves:
                    for kp in fc.keyframe_points:
                        all_frames.add(int(round(kp.co.x)))
            sk = getattr(obj.data, "shape_keys", None) if obj.data else None
            if sk and sk.animation_data and sk.animation_data.action:
                for fc in sk.animation_data.action.fcurves:
                    for kp in fc.keyframe_points:
                        all_frames.add(int(round(kp.co.x)))
        if not all_frames:
            return 0, args.num_frames - 1
        return min(all_frames), max(all_frames)

    def sample_frame_indices(frame_start, frame_end, num_frames, rng):
        """Motion324-style: random start + random skip."""
        T = frame_end - frame_start + 1
        if T < num_frames:
            indices = list(range(frame_start, frame_end + 1))
            while len(indices) < num_frames:
                indices.append(indices[-1])
            return indices[:num_frames]
        options = [{'skip': 1, 'span': num_frames, 'weight': 0.4}]
        span2 = (num_frames - 1) * 2 + 1
        if T >= span2:
            options.append({'skip': 2, 'span': span2, 'weight': 0.4})
        span4 = (num_frames - 1) * 4 + 1
        if T >= span4:
            options.append({'skip': 4, 'span': span4, 'weight': 0.2})
        total_w = sum(o['weight'] for o in options)
        probs = [o['weight'] / total_w for o in options]
        chosen = options[rng.choice(len(options), p=probs)]
        skip, span = chosen['skip'], chosen['span']
        max_start = T - span
        start = frame_start + (rng.randint(0, max_start + 1) if max_start > 0 else 0)
        return [start + i * skip for i in range(num_frames)]

    def get_global_bounds(sampled_frames):
        """Compute union AABB over sampled frames. Also tracks per-mesh
        extents (for free — same loop) to enable extent_ratio filtering.

        Vectorized: uses foreach_get + numpy matmul instead of a per-vertex
        Python loop. Equivalence to the original path verified in
        dataset_toolkits/test_vectorized_bounds.py (float32 precision)."""
        bbox_min = np.array([np.inf, np.inf, np.inf])
        bbox_max = np.array([-np.inf, -np.inf, -np.inf])
        per_mesh_min = {}
        per_mesh_max = {}
        for frame in sampled_frames:
            bpy.context.scene.frame_set(frame)
            bpy.context.view_layer.update()
            depsgraph = bpy.context.evaluated_depsgraph_get()
            for obj in bpy.context.scene.objects:
                if obj.type != 'MESH':
                    continue
                obj_eval = obj.evaluated_get(depsgraph)
                try:
                    mesh_eval = obj_eval.to_mesh()
                except Exception:
                    continue
                n = len(mesh_eval.vertices)
                if n == 0:
                    obj_eval.to_mesh_clear()
                    continue
                coords_local = np.empty(3 * n, dtype=np.float32)
                mesh_eval.vertices.foreach_get('co', coords_local)
                coords_local = coords_local.reshape(n, 3)
                M = np.array(obj_eval.matrix_world, dtype=np.float32)
                coords_world = coords_local @ M[:3, :3].T + M[:3, 3]
                mn = coords_world.min(axis=0)
                mx = coords_world.max(axis=0)
                bbox_min = np.minimum(bbox_min, mn)
                bbox_max = np.maximum(bbox_max, mx)
                name = obj.name
                if name in per_mesh_min:
                    per_mesh_min[name] = np.minimum(per_mesh_min[name], mn)
                    per_mesh_max[name] = np.maximum(per_mesh_max[name], mx)
                else:
                    per_mesh_min[name] = mn.copy()
                    per_mesh_max[name] = mx.copy()
                obj_eval.to_mesh_clear()
        center = (bbox_min + bbox_max) / 2
        scale = 1.0 / max(bbox_max - bbox_min)
        # Per-mesh max-axis extent
        per_mesh_extent = {}
        for name in per_mesh_min:
            per_mesh_extent[name] = float(np.max(per_mesh_max[name] - per_mesh_min[name]))
        return center, scale, per_mesh_extent

    def normalize_scene(sampled_frames, pre_bounds=None):
        """TRELLIS-style normalize with multi-frame union bbox for video.

        Wrap policy: wrap under ParentEmpty ONLY when there are multiple
        roots. Single root is scaled directly. This is TRELLIS's original
        logic (render.py:366-401) and is correct under Blender 3 where
        armature scaling propagates correctly through bone deformation.
        The NormEmpty-always-wrap workaround was a Blender 4 fix and is
        no longer needed after switching to Blender 3.

        Bbox source: get_global_bounds(sampled_frames), which evaluates
        the depsgraph at every sampled frame and accumulates a union AABB.
        This is the one change from TRELLIS's static pipeline that video
        processing requires — it ensures a single (scale, center) fits
        the entire motion envelope so all frames land in [-0.5, 0.5]^3.

        pre_bounds: optional (center, scale) cached from an earlier
        get_global_bounds(sampled_frames) call on the unmodified scene.
        Skips one redundant frame walk."""
        from mathutils import Vector
        scene_root_objects = [obj for obj in bpy.context.scene.objects.values() if not obj.parent]
        if not scene_root_objects:
            return np.zeros(3), 1.0
        if len(scene_root_objects) > 1:
            scene = bpy.data.objects.new("ParentEmpty", None)
            bpy.context.scene.collection.objects.link(scene)
            for obj in scene_root_objects:
                obj.parent = scene
        else:
            scene = scene_root_objects[0]
        if pre_bounds is not None:
            center, scale = pre_bounds
        else:
            center, scale, _ = get_global_bounds(sampled_frames)
        scene.scale = scene.scale * scale
        bpy.context.view_layer.update()
        center_after, _, _ = get_global_bounds(sampled_frames)
        offset = Vector((-center_after[0], -center_after[1], -center_after[2]))
        scene.matrix_world.translation += offset
        bpy.ops.object.select_all(action="DESELECT")
        return center, scale

    def sphere_hammersley(i, n, offset=(0, 0)):
        """TRELLIS utils.py sphere_hammersley_sequence."""
        def radical_inverse(base, n):
            val, inv_base, inv_base_n = 0, 1.0 / base, 1.0 / base
            while n > 0:
                val += (n % base) * inv_base_n
                n //= base
                inv_base_n *= inv_base
            return val
        u = i / n
        v = radical_inverse(2, i)
        u += offset[0] / n
        v += offset[1]
        u = 2 * u if u < 0.25 else 2 / 3 * u + 1 / 3
        theta = np.arccos(1 - 2 * u) - np.pi / 2
        phi = v * 2 * np.pi
        return phi, theta

    def generate_views(num_views, rng):
        """Multi-view set for DINOv2 feature extraction.
        Matches TRELLIS render.py: sphere_hammersley with random offset,
        fixed radius=2, fov=40 deg."""
        offset = (rng.rand(), rng.rand())
        views = []
        for i in range(num_views):
            yaw, pitch = sphere_hammersley(i, num_views, offset)
            views.append({'yaw': float(yaw), 'pitch': float(pitch),
                          'radius': 2.0, 'fov': 40.0 / 180.0 * math.pi})
        return views

    def sample_radius_fov(rng):
        """TRELLIS render_cond.py distribution: uniform in k=1/r^2,
        fov derived so projected size of unit cube stays constant.
        fov in [10, 70] deg."""
        fov_min, fov_max = 10.0, 70.0
        radius_min = math.sqrt(3) / 2 / math.sin(math.radians(fov_max) / 2)
        radius_max = math.sqrt(3) / 2 / math.sin(math.radians(fov_min) / 2)
        k_min = 1.0 / radius_max ** 2
        k_max = 1.0 / radius_min ** 2
        k = rng.uniform(k_min, k_max)
        radius = 1.0 / math.sqrt(k)
        fov = 2 * math.asin(math.sqrt(3) / 2 / radius)
        return float(radius), float(fov)

    def generate_cond_views(n, rng):
        """N cond cameras for video conditioning (held constant across a clip).
        view_00 is always the front view (yaw=3pi/2, pitch=10 deg) with
        TRELLIS-sampled radius/fov. Remaining N-1 views are sphere_hammersley
        with random offset, each with independently sampled radius/fov."""
        views = []
        r0, fov0 = sample_radius_fov(rng)
        views.append({'yaw': 3 * math.pi / 2, 'pitch': math.radians(10),
                      'radius': r0, 'fov': fov0})
        if n > 1:
            offset = (rng.rand(), rng.rand())
            for i in range(n - 1):
                yaw, pitch = sphere_hammersley(i, n - 1, offset)
                r, fov = sample_radius_fov(rng)
                views.append({'yaw': float(yaw), 'pitch': float(pitch),
                              'radius': r, 'fov': fov})
        return views

    # ---- Main render logic ----
    run_id = f"run_{args.seed}" if args.add else ""

    # Reproducible global RNG (in case any unseeded numpy call sneaks in).
    np.random.seed(args.seed + args.rank)

    glb_files = sorted([f for f in os.listdir(args.glb_dir) if f.endswith('.glb')])
    if args.max_objects > 0:
        glb_files = glb_files[:args.max_objects]
    total = len(glb_files)
    start = total * args.rank // args.world_size
    end = total * (args.rank + 1) // args.world_size
    my_files = glb_files[start:end]

    import time as _time

    def get_done_flag(obj_id):
        if run_id:
            return os.path.join(args.output_dir, 'render_done', f'{obj_id}__{run_id}.done')
        return os.path.join(args.output_dir, 'render_done', f'{obj_id}.done')

    def get_output_subdir(base, obj_id):
        if run_id:
            return os.path.join(base, f'{obj_id}__{run_id}')
        return os.path.join(base, obj_id)

    skipped = sum(1 for f in my_files if os.path.exists(get_done_flag(os.path.splitext(f)[0])))

    log_progress(args, f"[R{args.rank}] Render: {len(my_files)} objects ({start}-{end} of {total}), "
          f"{args.num_frames} frames x {args.num_views} views")

    success, fail, processed = 0, 0, 0
    t_start = _time.time()
    todo = len(my_files) - skipped

    for file_idx, glb_file in enumerate(my_files):
        obj_id = os.path.splitext(glb_file)[0]
        done_flag = get_done_flag(obj_id)
        if os.path.exists(done_flag):
            success += 1
            continue

        # Stable hash (Python's hash() is randomized per process unless
        # PYTHONHASHSEED is set, which would break reproducibility).
        obj_seed = (args.seed + zlib.adler32(obj_id.encode('utf-8'))) % (2 ** 31)
        obj_rng = np.random.RandomState(obj_seed)

        # Per-clip cameras: sampled inside the loop so each object gets its
        # own trajectory, but shared across all frames within the clip.
        views = generate_views(args.num_views, obj_rng)
        cond_views = generate_cond_views(args.num_cond_views, obj_rng)

        try:
            clear_scene()
            bpy.ops.import_scene.gltf(filepath=os.path.join(args.glb_dir, glb_file))
            remove_junk_objects()
            setup_renderer(args.resolution)

            frame_start, frame_end = get_animation_frame_range()
            sampled_frames = sample_frame_indices(frame_start, frame_end, args.num_frames, obj_rng)
            actual_frames = len(sampled_frames)

            # Pre-normalization quality check: compute per-mesh extents
            # (free — piggybacked on get_global_bounds's depsgraph walk)
            # and skip if one submesh dominates by >15x (the character
            # would be crushed to <100 voxels). Cache (center, scale) to
            # feed into normalize_scene and avoid a redundant frame walk.
            pre_center, pre_scale, per_mesh_extent = get_global_bounds(sampled_frames)
            if len(per_mesh_extent) > 1:
                extents = list(per_mesh_extent.values())
                max_ext = max(extents)
                median_ext = float(np.median(extents))
                if median_ext > 0 and max_ext / median_ext > args.max_extent_ratio:
                    worst = max(per_mesh_extent, key=per_mesh_extent.get)
                    msg = (f"[R{args.rank}] SKIP {obj_id}: "
                           f"extent_ratio={max_ext/median_ext:.1f}x "
                           f"(worst: {worst}={max_ext:.1f})")
                    log_progress(args, msg, is_error=True)
                    # Record skipped sample
                    skip_path = os.path.join(args.output_dir, 'skipped_render.txt')
                    with open(skip_path, 'a') as sf:
                        sf.write(f"{obj_id}\textent_ratio={max_ext/median_ext:.1f}x\n")
                    fail += 1
                    processed += 1
                    continue

            # Normalize BEFORE lights/camera (TRELLIS order).
            # Reuse the pre-normalization bounds computed above for the
            # extent-ratio filter — scene hasn't been modified since, so
            # the pre-scale (center, scale) is identical to what
            # normalize_scene would recompute on its first call.
            global_center, global_scale = normalize_scene(
                sampled_frames, pre_bounds=(pre_center, pre_scale)
            )
            setup_lighting()
            cam, cam_empty = init_camera(track_target=(0, 0, 0))

            # Per-object output directories (constant across frames).
            obj_render_dir = get_output_subdir(os.path.join(args.output_dir, 'renders'), obj_id)
            obj_mesh_dir = get_output_subdir(os.path.join(args.output_dir, 'meshes'), obj_id)
            obj_cond_dir = get_output_subdir(os.path.join(args.output_dir, 'renders_cond'), obj_id)
            os.makedirs(obj_mesh_dir, exist_ok=True)
            os.makedirs(obj_cond_dir, exist_ok=True)
            for ci in range(len(cond_views)):
                os.makedirs(os.path.join(obj_cond_dir, f'view_{ci:02d}'), exist_ok=True)

            # Precompute cond camera metadata once — cameras are held
            # constant across the clip, only the animated geometry changes.
            cond_frames_meta = []
            for ci, cview in enumerate(cond_views):
                cam.location = (
                    cview['radius'] * math.cos(cview['yaw']) * math.cos(cview['pitch']),
                    cview['radius'] * math.sin(cview['yaw']) * math.cos(cview['pitch']),
                    cview['radius'] * math.sin(cview['pitch']),
                )
                cam.data.lens = 16 / math.tan(cview['fov'] / 2)
                bpy.context.view_layer.update()
                cond_frames_meta.append({
                    'view': ci,
                    'camera_angle_x': cview['fov'],
                    'transform_matrix': get_transform_matrix(cam),
                })
            cond_transforms = {
                "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                "scale": float(global_scale),
                "offset": global_center.tolist(),
                "num_cond_views": int(args.num_cond_views),
                "views": cond_frames_meta,
            }
            with open(os.path.join(obj_cond_dir, 'transforms.json'), 'w') as f:
                json.dump(cond_transforms, f, indent=2)

            for t, anim_frame in enumerate(sampled_frames):
                frame_render_dir = os.path.join(obj_render_dir, f'frame_{t:04d}')
                mesh_path = os.path.join(obj_mesh_dir, f'frame_{t:04d}.ply')
                os.makedirs(frame_render_dir, exist_ok=True)

                bpy.context.scene.frame_set(anim_frame)
                bpy.context.view_layer.update()
                # TRELLIS-style direct operator export (Blender 3).
                # Select only MESH objects to avoid "Object does not have
                # geometry data" warnings from LIGHT/CAMERA/EMPTY/ARMATURE.
                bpy.ops.object.select_all(action='DESELECT')
                for obj in bpy.context.scene.objects:
                    if obj.type == 'MESH':
                        obj.select_set(True)
                bpy.ops.export_mesh.ply(filepath=mesh_path, use_selection=True)

                to_export = {
                    "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                    "scale": float(global_scale),
                    "offset": global_center.tolist(),
                    "frames": []
                }

                for vi, view in enumerate(views):
                    cam.location = (
                        view['radius'] * math.cos(view['yaw']) * math.cos(view['pitch']),
                        view['radius'] * math.sin(view['yaw']) * math.cos(view['pitch']),
                        view['radius'] * math.sin(view['pitch']),
                    )
                    cam.data.lens = 16 / math.tan(view['fov'] / 2)
                    bpy.context.view_layer.update()
                    img_path = os.path.join(frame_render_dir, f'{vi:03d}.png')
                    bpy.context.scene.render.filepath = img_path
                    bpy.ops.render.render(write_still=True)
                    to_export["frames"].append({
                        "file_path": f'{vi:03d}.png',
                        "camera_angle_x": view['fov'],
                        "transform_matrix": get_transform_matrix(cam),
                    })

                with open(os.path.join(frame_render_dir, 'transforms.json'), 'w') as f:
                    json.dump(to_export, f, indent=2)

                # Per-frame conditioning images: same cameras as above,
                # only the animated geometry differs between frames.
                for ci, cview in enumerate(cond_views):
                    cam.location = (
                        cview['radius'] * math.cos(cview['yaw']) * math.cos(cview['pitch']),
                        cview['radius'] * math.sin(cview['yaw']) * math.cos(cview['pitch']),
                        cview['radius'] * math.sin(cview['pitch']),
                    )
                    cam.data.lens = 16 / math.tan(cview['fov'] / 2)
                    bpy.context.view_layer.update()
                    cond_img_path = os.path.join(obj_cond_dir, f'view_{ci:02d}', f'frame_{t:04d}.png')
                    bpy.context.scene.render.filepath = cond_img_path
                    bpy.ops.render.render(write_still=True)

            os.makedirs(os.path.join(args.output_dir, 'render_done'), exist_ok=True)
            with open(done_flag, 'w') as f:
                json.dump({
                    'obj_id': obj_id, 'num_frames': actual_frames,
                    'sampled_frames': sampled_frames,
                    'global_center': global_center.tolist(),
                    'global_scale': float(global_scale),
                    'seed': args.seed, 'run_id': run_id,
                }, f)
            success += 1
            processed += 1
        except Exception as e:
            log_progress(args, f"[R{args.rank}] ERROR {obj_id}: {e}", is_error=True)
            fail += 1
            processed += 1

        elapsed = _time.time() - t_start
        if processed > 0:
            avg_sec = elapsed / processed
            eta = (todo - processed) * avg_sec
            eta_str = f"{eta/3600:.1f}h" if eta > 3600 else f"{eta/60:.0f}m"
        else:
            avg_sec, eta_str = 0, "?"
        update_status(args, f"[R{args.rank} render] {processed}/{todo} | ok={success} fail={fail} | {avg_sec:.0f}s/obj | ETA {eta_str}")

    update_status(args, f"[R{args.rank} render] DONE {success} ok, {fail} fail | {(_time.time()-t_start)/3600:.1f}h")


# ============================================================================
# Pass 2: PyTorch encoding
# ============================================================================

def pass_encode(args):
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    from PIL import Image

    # Add TRELLIS to path
    trellis_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, trellis_root)

    import trellis.models as models
    import trellis.modules.sparse as sp

    torch.set_grad_enabled(False)
    device = torch.device(f'cuda:{args.rank % torch.cuda.device_count()}')

    # Load models
    print(f"[R{args.rank}] Loading models on {device}...")
    dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
    dinov2_model.eval().to(device)
    dino_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    n_patch = 518 // 14

    ss_encoder = models.from_pretrained('JeffreyXiang/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16')
    ss_encoder.eval().to(device)
    slat_encoder = models.from_pretrained('JeffreyXiang/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16')
    slat_encoder.eval().to(device)
    print(f"[R{args.rank}] Models loaded.")

    import open3d as o3d
    import utils3d

    def voxelize_mesh(mesh_path, resolution=64):
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        vertices = np.clip(np.asarray(mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
            mesh, voxel_size=1.0 / resolution,
            min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
        indices = np.array([v.grid_index for v in voxel_grid.get_voxels()])
        if len(indices) == 0:
            return np.zeros((1, resolution, resolution, resolution), dtype=np.float32), \
                   np.zeros((0, 3), dtype=np.int64), np.zeros((0, 3), dtype=np.float64)
        assert np.all(indices >= 0) and np.all(indices < resolution)
        positions = (indices + 0.5) / resolution - 0.5
        ss = np.zeros((1, resolution, resolution, resolution), dtype=np.float32)
        ss[0, indices[:, 0], indices[:, 1], indices[:, 2]] = 1.0
        return ss, indices, positions

    def load_frame_renders(obj_id, frame_t):
        render_dir = os.path.join(args.output_dir, 'renders', obj_id, f'frame_{frame_t:04d}')
        with open(os.path.join(render_dir, 'transforms.json'), 'r') as f:
            meta = json.load(f)
        data = []
        for frame_info in meta['frames']:
            img_path = os.path.join(render_dir, frame_info['file_path'])
            try:
                image = Image.open(img_path).resize((518, 518), Image.Resampling.LANCZOS)
            except Exception:
                continue
            image = np.array(image).astype(np.float32) / 255
            image_rgb = image[:, :, :3] * image[:, :, 3:]
            image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float()
            image_tensor = dino_transform(image_tensor)
            c2w = torch.tensor(frame_info['transform_matrix']).float()
            c2w[:3, 1:3] *= -1
            extrinsics = torch.inverse(c2w)
            fov = frame_info['camera_angle_x']
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(torch.tensor(fov), torch.tensor(fov))
            data.append({'image': image_tensor, 'extrinsics': extrinsics, 'intrinsics': intrinsics})
        return data

    def extract_features(data, positions, batch_size=16):
        positions_t = torch.from_numpy(positions).float().to(device)
        indices = ((positions_t + 0.5) * 64).long()
        patchtokens_all, uv_all = [], []
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            bs = len(batch)
            images = torch.stack([d['image'] for d in batch]).to(device)
            extrinsics = torch.stack([d['extrinsics'] for d in batch]).to(device)
            intrinsics = torch.stack([d['intrinsics'] for d in batch]).to(device)
            features = dinov2_model(images, is_training=True)
            uv = utils3d.torch.project_cv(positions_t, extrinsics, intrinsics)[0] * 2 - 1
            patchtokens = features['x_prenorm'][:, dinov2_model.num_register_tokens + 1:]
            patchtokens = patchtokens.permute(0, 2, 1).reshape(bs, 1024, n_patch, n_patch)
            patchtokens_all.append(patchtokens)
            uv_all.append(uv)
        patchtokens = torch.cat(patchtokens_all, dim=0)
        uv = torch.cat(uv_all, dim=0)
        sampled = F.grid_sample(patchtokens, uv.unsqueeze(1), mode='bilinear', align_corners=False)
        sampled = sampled.squeeze(2).permute(0, 2, 1)
        mean_features = sampled.mean(dim=0).cpu().numpy().astype(np.float16)
        return indices.cpu().numpy().astype(np.uint8), mean_features

    def encode_ss(voxel_grid):
        ss_tensor = torch.from_numpy(voxel_grid).float().unsqueeze(0).to(device)
        latent = ss_encoder(ss_tensor, sample_posterior=False)
        return latent[0].cpu().numpy()

    def encode_slat(indices, features):
        feats_t = torch.from_numpy(features).float()
        coords_t = torch.cat([torch.zeros(features.shape[0], 1).int(),
                              torch.from_numpy(indices).int()], dim=1)
        sparse_input = sp.SparseTensor(feats=feats_t, coords=coords_t).to(device)
        latent = slat_encoder(sparse_input, sample_posterior=False)
        return {
            'feats': latent.feats.cpu().numpy().astype(np.float32),
            'coords': latent.coords[:, 1:].cpu().numpy().astype(np.uint8),
        }

    # ---- Main encoding loop ----
    render_done_dir = os.path.join(args.output_dir, 'render_done')
    encode_done_dir = os.path.join(args.output_dir, 'encode_done')
    os.makedirs(encode_done_dir, exist_ok=True)

    def get_pending_objects():
        if not os.path.exists(render_done_dir):
            return []
        rendered = set(f.replace('.done', '') for f in os.listdir(render_done_dir) if f.endswith('.done'))
        encoded = set(f.replace('.done', '') for f in os.listdir(encode_done_dir) if f.endswith('.done'))
        pending = sorted(rendered - encoded)
        if args.max_objects > 0:
            glb_files = sorted([os.path.splitext(f)[0] for f in os.listdir(args.glb_dir) if f.endswith('.glb')])
            allowed = set(glb_files[:args.max_objects])
            pending = [p for p in pending if p.split('__run_')[0] in allowed]
        total = len(pending)
        s = total * args.rank // args.world_size
        e = total * (args.rank + 1) // args.world_size
        return pending[s:e]

    import time as _time
    poll_mode = getattr(args, 'poll', False)
    idle_rounds = 0
    max_idle_rounds = 30

    ss_latent_dir = os.path.join(args.output_dir, 'ss_latents')
    slat_latent_dir = os.path.join(args.output_dir, 'slat_latents')
    os.makedirs(ss_latent_dir, exist_ok=True)
    os.makedirs(slat_latent_dir, exist_ok=True)

    def encode_one_object(obj_id):
        obj_ss_dir = os.path.join(ss_latent_dir, obj_id)
        obj_slat_dir = os.path.join(slat_latent_dir, obj_id)
        encode_done_path = os.path.join(encode_done_dir, f'{obj_id}.done')
        if os.path.exists(encode_done_path):
            return True
        with open(os.path.join(render_done_dir, f'{obj_id}.done'), 'r') as f:
            meta = json.load(f)
        num_frames = meta['num_frames']
        os.makedirs(obj_ss_dir, exist_ok=True)
        os.makedirs(obj_slat_dir, exist_ok=True)
        # First pass: voxelize all frames and check min voxel count.
        # If ANY frame has < min_voxels, skip the entire GLB.
        all_voxel_data = []
        for t in range(num_frames):
            mesh_path = os.path.join(args.output_dir, 'meshes', obj_id, f'frame_{t:04d}.ply')
            voxel_grid, indices, positions = voxelize_mesh(mesh_path)
            all_voxel_data.append((voxel_grid, indices, positions))
            if indices.shape[0] < args.min_voxels:
                print(f"[R{args.rank}] SKIP {obj_id}: frame {t} has "
                      f"{indices.shape[0]} voxels < {args.min_voxels}")
                skip_path = os.path.join(args.output_dir, 'skipped_encode.txt')
                with open(skip_path, 'a') as sf:
                    sf.write(f"{obj_id}\tframe_{t}_voxels={indices.shape[0]}\n")
                return False

        # Second pass: encode (all frames passed the voxel check)
        for t in range(num_frames):
            voxel_grid, indices, positions = all_voxel_data[t]
            if indices.shape[0] == 0:
                continue
            ss_latent = encode_ss(voxel_grid)
            np.savez_compressed(os.path.join(obj_ss_dir, f'frame_{t:04d}.npz'), mean=ss_latent)
            render_data = load_frame_renders(obj_id, t)
            if len(render_data) == 0:
                continue
            feat_indices, feat_tokens = extract_features(render_data, positions)
            slat_pack = encode_slat(feat_indices, feat_tokens)
            np.savez_compressed(os.path.join(obj_slat_dir, f'frame_{t:04d}.npz'), **slat_pack)
        if not args.keep_intermediates:
            for d in [os.path.join(args.output_dir, 'renders', obj_id),
                      os.path.join(args.output_dir, 'meshes', obj_id)]:
                if os.path.exists(d):
                    shutil.rmtree(d)
        with open(encode_done_path, 'w') as f:
            f.write('done\n')
        return True

    success, fail = 0, 0
    t_start = _time.time()
    initial_pending = get_pending_objects()
    total_seen = len(initial_pending)
    log_progress(args, f"[R{args.rank}] Encode: {total_seen} pending{'  (poll)' if poll_mode else ''}")

    while True:
        pending = get_pending_objects()
        if len(pending) == 0:
            if not poll_mode:
                break
            idle_rounds += 1
            if idle_rounds >= max_idle_rounds:
                break
            _time.sleep(10)
            continue
        idle_rounds = 0
        total_seen = max(total_seen, success + fail + len(pending))
        for obj_id in pending:
            try:
                encode_one_object(obj_id)
                success += 1
            except Exception as e:
                import traceback
                log_progress(args, f"[R{args.rank}] ERROR {obj_id}: {e}\n{traceback.format_exc()}", is_error=True)
                fail += 1
            done = success + fail
            elapsed = _time.time() - t_start
            avg = elapsed / done if done > 0 else 0
            remaining = total_seen - done
            eta = remaining * avg
            eta_str = f"{eta/3600:.1f}h" if eta > 3600 else f"{eta/60:.0f}m"
            update_status(args, f"[R{args.rank} encode] {done}/{total_seen} | ok={success} fail={fail} | {avg:.1f}s/obj | ETA {eta_str}")
        if not poll_mode:
            break

    update_status(args, f"[R{args.rank} encode] DONE {success} ok, {fail} fail | {(_time.time()-t_start)/3600:.1f}h")


# ============================================================================
# Main
# ============================================================================

def parse_args():
    argv = sys.argv
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]
    else:
        argv = argv[1:]
    parser = argparse.ArgumentParser(description='Prepare dynamic mesh data for TRELLIS training')
    parser.add_argument('--pass', dest='pipeline_pass', type=str, required=True, choices=['render', 'encode'])
    parser.add_argument('--glb_dir', type=str, required=True,
                        help='Directory of animated .glb files to process.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory where the prepared dataset tree will be written.')
    parser.add_argument('--num_frames', type=int, default=12)
    parser.add_argument('--num_views', type=int, default=60)
    parser.add_argument('--num_cond_views', type=int, default=6,
                        help='Number of conditioning views per clip (view_00 is always front).')
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--keep_intermediates', action='store_true')
    parser.add_argument('--poll', action='store_true')
    parser.add_argument('-n', '--max_objects', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--add', action='store_true')
    parser.add_argument('--max_extent_ratio', type=float, default=15.0,
                        help='Skip GLB if any submesh extent > this × median (render pass).')
    parser.add_argument('--min_voxels', type=int, default=500,
                        help='Skip GLB if any frame has fewer voxels than this (encode pass).')
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.pipeline_pass == 'render':
        pass_render(args)
    elif args.pipeline_pass == 'encode':
        pass_encode(args)
