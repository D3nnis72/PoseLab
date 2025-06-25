#!/usr/bin/env python3
"""
Convert custom 6DOF dataset into BOP-M format with proper camera.json, test_targets_bop19.json,
plus aggregated scene_camera.json, scene_gt.json, and scene_gt_info.json in the test folder.

python convert_to_bopm.py --input_dir "KurbelGehauseOben" --output_dir "lmo"
"""
import os
import shutil
import json
from pathlib import Path
import numpy as np

def read_intrinsics_txt(path):
    with open(path, encoding='utf-8') as f:
        tokens = f.read().split()
    vals = [t.replace(',', '.') for t in tokens]
    K = np.array(vals, dtype=float).reshape(3, 3)
    return K


def gather_scenes(input_dir):
    return sorted([d for d in Path(input_dir).iterdir() if d.is_dir()], key=lambda x: int(x.name))


def ensure_folder(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def main(args):
    input_dir = Path(args.input_dir)
    out = Path(args.output_dir)
    scenes = gather_scenes(input_dir)

    # Prepare output dirs
    models = out / 'models'; ensure_folder(models)
    models_eval = out / 'models_eval'; ensure_folder(models_eval)
    test_dir = out / 'test'; ensure_folder(test_dir)


    obj_id_str = '000001'

    global_K = None
    depth_scale = 1.0
    all_test_images = []
    gt_entries = {}
    gt_info_entries = {}
    all_scene_cams = {}

    # Process each scene
    for scene in scenes:
        sid = int(scene.name)
        sid_str = f"{sid:06d}"
        scene_out = test_dir / sid_str
        ensure_folder(scene_out)

        # 1) Copy rgb, depth, mask folders
        for mod in ('rgb', 'depth', 'masks'):
            src = scene / 'FoundationPose' / mod
            dst_name = 'mask' if mod == 'masks' else mod
            dst = scene_out / dst_name
            ensure_folder(dst)
            for img in sorted(src.glob('*.png')):
                shutil.copy(img, dst / img.name)

        # 2) Read per-scene intrinsics once, write scene_out/camera.json
        K_scene = read_intrinsics_txt(scene / 'FoundationPose' / 'cam_K.txt')
        fx, fy = float(K_scene[0, 0]), float(K_scene[1, 1])
        cx, cy = float(K_scene[0, 2]), float(K_scene[1, 2])

        camera_scene = {
                "cam_K": K_scene.flatten().tolist(),
                "depth_scale": depth_scale
        }
        (scene_out / 'camera.json').write_text(json.dumps(camera_scene, indent=2))

        img_ids = [int(p.stem) for p in (scene / 'FoundationPose' / 'rgb').glob('*.png')]
        this_cam = {}
        for im_id in img_ids:
            im_str = f"{im_id:06d}"
            cam_txt = scene / 'FoundationPose' / 'cam_K.txt'
            if not cam_txt.exists():
                raise FileNotFoundError(f"Missing {cam_txt}")
            K_im = read_intrinsics_txt(cam_txt)
            this_cam[str(im_id)] = {
                "cam_K": K_im.flatten().tolist(),
                "depth_scale": depth_scale
            }
        (scene_out / 'scene_camera.json').write_text(json.dumps(this_cam, indent=2))

        # Accumulate for top-level scene_camera.json
        all_scene_cams[sid_str] = this_cam

        # Populate GT placeholders and test_targets list
        for im_id in img_ids:
            all_test_images.append((sid, im_id))
            gt_entries.setdefault(str(im_id), []).append({
                "cam_R_m2c": [1,0,0,0,1,0,0,0,1],
                "cam_t_m2c": [0.0,0.0,0.0],
                "obj_id": 1
            })
            gt_info_entries.setdefault(str(im_id), []).append({
                "bbox_obj": [0,0,1,1],
                "bbox_visib": [0,0,1,1],
                "px_count_all": 1,
                "px_count_valid": 1,
                "px_count_visib": 1,
                "visib_fract": 1.0
            })

    # Write aggregated per-scene file under test_dir/scene_camera.json
    (test_dir / 'scene_camera.json').write_text(json.dumps(all_scene_cams, indent=2))

    # Write global test_dir/camera.json using first scene intrinsics
    first_scene = scenes[0]
    K0 = read_intrinsics_txt(first_scene / 'FoundationPose' / 'cam_K.txt')
    fx0, fy0 = float(K0[0,0]), float(K0[1,1])
    cx0, cy0 = float(K0[0,2]), float(K0[1,2])
    global_camera = {
        "fx": fx0, "fy": fy0,
        "cx": cx0, "cy": cy0,
        "depth_scale": depth_scale,
        "height": 480, "width": 640
    }
    (test_dir / 'camera.json').write_text(json.dumps(global_camera, indent=2))

    # Write test_targets_bop19.json and other files as before
    test_targets = [
        {"scene_id": sid, "im_id": im_id, "inst_count": len(gt_entries[str(im_id)]), "obj_id": 1}
        for sid, im_id in all_test_images
    ]
    (out / 'test_targets_bop19.json').write_text(json.dumps(test_targets, indent=2))

    scene_gt_src = input_dir / '0' / 'GT' / 'scene_gt.json'
    scene_gt_info_src = input_dir / '0' / 'GT' / 'scene_gt_info.json'

    scene_gt_out = test_dir / '000000' / 'scene_gt.json'
    scene_gt_info_out = test_dir / '000000' / 'scene_gt_info.json'
    shutil.copy(scene_gt_src, scene_gt_out)
    shutil.copy(scene_gt_info_src, scene_gt_info_out)


    print("Conversion to BOP-M complete.")

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--input_dir', required=True)
    p.add_argument('--output_dir', required=True)
    args = p.parse_args()

    main(args)
