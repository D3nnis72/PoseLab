#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
import time
import json
import numpy as np
from pathlib import Path

def run(cmd, cwd=None):
    print(f"[run] {' '.join(cmd)} (cwd={cwd or os.getcwd()})")
    res = subprocess.run(cmd, cwd=cwd)
    if res.returncode != 0:
        print(f"[error] Command failed: {' '.join(cmd)}", file=sys.stderr)
        sys.exit(res.returncode)

def copy_tree(src, dst):
    if not os.path.isdir(src):
        print(f"[warn] Source not found: {src}, skipping copy")
        return
    print(f"[copy] {src} → {dst}")
    shutil.copytree(src, dst, dirs_exist_ok=True)

def copy_file(src, dst):
    if not os.path.isfile(src):
        print(f"[warn] File not found: {src}, skipping")
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    print(f"[copy] {src} → {dst}")
    shutil.copy(src, dst)

def copy_if_exists(src, dst, is_dir=False):
    if is_dir:
        if os.path.isdir(src):
            print(f"Copying tree {src} → {dst}")
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            print(f"[warning] missing dir {src}, skipping copy")
    else:
        if os.path.isfile(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            print(f"Copying file {src} → {dst}")
            shutil.copy(src, dst)
        else:
            print(f"[warning] missing file {src}, skipping copy")

def read_intrinsics_txt(path):
    with open(path, encoding='utf-8') as f:
        tokens = f.read().split()
    vals = [t.replace(',', '.') for t in tokens]
    K = np.array(vals, dtype=float).reshape(3, 3)
    return K

def copy_model_assets(raw_models: Path, object_type: str, gigapose_root: Path):
    src_model_dir = raw_models / object_type / "model"
    for fname, subdir in [("models_info.json", "datasets/lmo/models"),
                          ("models_info.json", "datasets/lmo/models_eval"),
                          ("obj_000001.ply",   "datasets/lmo/models"),
                          ("obj_000001.ply",   "datasets/lmo/models_eval")]:
        src = src_model_dir / fname
        dst = gigapose_root / subdir / fname
        copy_if_exists(str(src), str(dst))

def copy_test_data(example_root: Path, gigapose_root: Path, scene="000000", exp_gt='', exp_gt_info=''):
    lmo_root = gigapose_root / "datasets" / "lmo" / "test" / scene
    lmo_root.mkdir(parents=True, exist_ok=True)

    # simple mapping of src→dst rel paths
    files = {
        example_root / "outputs" / "predictions.csv":
            gigapose_root / "results/large_000000/predictions/large-pbrreal-rgb-mmodel_lmo-test_000000.csv",
        example_root / "SAM6D" / "rgb.png":
            lmo_root / "rgb" / f"{scene}.png",
        example_root / "SAM6D" / "depth.png":
            lmo_root / "depth" / f"{scene}.png",
        example_root / "SAM6D" / "camera.json":
            lmo_root / "camera.json",
        exp_gt:
            lmo_root / "scene_gt.json",
        exp_gt_info:
            lmo_root / "scene_gt_info.json",
    }

    for src, dst in files.items():
        copy_if_exists(str(src), str(dst))

def write_scene_camera(export_root: Path, gigapose_root: Path, scene="000000"):
    rgb_dir = export_root / "FoundationPose" / "rgb"
    cam_txt = export_root / "FoundationPose" / "cam_K.txt"

    if not rgb_dir.is_dir() or not cam_txt.exists():
        print(f"[warn] Missing FoundationPose data at {rgb_dir} or {cam_txt}, skipping scene_camera.json")
        return

    # load and flatten intrinsics once
    K = read_intrinsics_txt(str(cam_txt)).flatten().tolist()

    # build camera dict from whatever PNGs we find
    cams = {
        p.stem: {"cam_K": K, "depth_scale": 1.0}
        for p in rgb_dir.glob("*.png")
    }

    out = gigapose_root / "datasets" / "lmo" 
    out.parent.mkdir(parents=True, exist_ok=True)
    (out / "test" / scene  / "scene_camera.json").write_text(json.dumps(cams, indent=2))


    targets = [
        {
            "scene_id": int(scene),
            "im_id": 0,
            "inst_count": 1,
            "obj_id": 1,
        }
    ]
    (out / "test_targets_bop19.json").write_text(json.dumps(targets, indent=2))
 

def main():
    p = argparse.ArgumentParser(
        description="Pipeline for SAM6D model inference on a single capture"
    )
    p.add_argument("--export_root", default="../SharedData/WebsocketData/000000/0/",   required=False,
                   help="Folder containing SAM6D raw: e.g. ./WebsocketData/000000/0/")
    p.add_argument("--example_root", default="./Data/Example",  required=False,
                   help="Target folder for this run: e.g. Data/Example")
    p.add_argument("--cad", default="obj_000001.ply",     required=False,
                   help="Name of your CAD file under example_root, e.g. obj_000005.ply")
    p.add_argument("--object_type", default="crankcase",   required=False,
                   help="e.g. crankcase (used to copy pre-existing models if any)")
    p.add_argument("--raw_models", default="../SharedData/ObjectModels",
                   help="Root of your object_models folder")
    p.add_argument("--novel",     default=False,
                   help="If its an new object")
    p.add_argument("--scene_name",     default="Default")
    p.add_argument("--visibility",     default="Visibility")
    p.add_argument("--lighting",     default="Lighting")
    p.add_argument("--distance",     default="Distance")
    p.add_argument("--height",     default="Height")
    p.add_argument("--test_all",     default=False,
                   help="Test all Pipelines")
    p.add_argument("--test_all_primary", default=True)
    p.add_argument("--primary_result_file")

    args = p.parse_args()

    timings = {}

    global_start = time.time()

    if args.test_all_primary == False or args.test_all_primary == "False":
        print(f"primary result: {args.primary_result_file}")
        timings_file = os.path.join(args.primary_result_file, 'Scores' , 'timings.json')
        print(timings_file)
        if args.primary_result_file and os.path.isfile(timings_file):
            print("Found")
            with open(timings_file, "r") as f:
              timings = json.load(f)

    exp_sam = args.export_root
    dst = args.example_root

    cad_ply = os.path.join(dst, args.cad)

    # 1) Copy photos
    print("\n==> Step 1: Copy SAM6D photos")
    timings['copy_input_start'] = time.time()
    copy_tree(exp_sam, dst)

    # Dein Pfad
    from PIL import Image
    rgb_image = os.path.join(dst, "FoundationPose", "rgb", "000000.png")
    img = Image.open(rgb_image).convert("RGB")
    img.save(rgb_image)


    print("\n==> Step 2: Copy precomputed model info if present")
    src_model_dir = os.path.join(args.raw_models, args.object_type, "model")
    dst_model_dir = dst 
    copy_tree(src_model_dir, dst_model_dir)

    src_model_dir = os.path.join(args.raw_models, args.object_type, "model", "obj_000001.ply")
    dst_foundationpose_model_dir = os.path.join(dst, "FoundationPose", "mesh", "obj_000001.ply")
    copy_file(src_model_dir, dst_foundationpose_model_dir)
    
    exp_gt = os.path.join(args.export_root, "GT", "scene_gt.json")
    exp_gt_info = os.path.join(args.export_root, "GT", "scene_gt_info.json")

    # 2) Render Templates (if new)
    timings['copy_input_end'] = time.time()

    if args.test_all_primary == True:
        timings['copy_templates_start'] = time.time()
        if args.novel:
            print("\n==> Step 3: Render templates via BlenderProc")
            # pass --normalize flag if needed
            run([
                "blenderproc", "run", "render/render_templates_optimized.py",
                "--output_dir", os.path.join(dst, "outputs"),
                "--cad_path",    cad_ply,
                "--normalize"
            ], cwd="Render")
        else:
            print("\n==> Step 3: Copy precomputed segmentation templates")
            src_model_dir = os.path.join(args.raw_models, args.object_type, "segmentation_templates")
            dst_model_dir = os.path.join(dst, "SAM6D", "outputs")
            copy_tree(src_model_dir, dst_model_dir)
        timings['copy_templates_end'] = time.time()

    # 4) Render Instance Segmentation
    if args.test_all_primary == True:
        print("\n==> Step 4: Instance segmentation with SAM")
        timings['instance_segmentation_start'] = time.time()
        run([
            "python", "run_inference_custom.py",
            "--segmentor_model", "sam",
            "--output_dir",       os.path.join('..',dst, 'SAM6D', "outputs"),
            "--cad_path",         os.path.join('..', cad_ply),
            "--rgb_path",         os.path.join('..',dst, 'SAM6D',"rgb.png"),
            "--depth_path",       os.path.join('..',dst, 'SAM6D',"depth.png"),
            "--cam_path",         os.path.join('..', dst, 'SAM6D',"camera.json"),
        ], cwd="Instance_Segmentation_Model")
        timings['instance_segmentation_start'] = time.time()
    else:
        print("\n==> Step 4: Copy Segmentation Templates")
        ism_result_file = os.path.join(args.primary_result_file, 'Pose','InstanceSegmentation.json')
        dest_ism_result_file =  "./Data/Example/SAM6D/outputs/sam6d_results/detection_ism.json"
        copy_if_exists(ism_result_file, dest_ism_result_file)

        ism_mask_file = os.path.join(args.primary_result_file, 'Pose','MaskISM.png')
        dest_ism_mask_file =  "./Data/Example/SAM6D/outputs/sam6d_results/mask_ism.png"
        copy_if_exists(ism_mask_file, dest_ism_mask_file)

        
        ism_vis_file = os.path.join(args.primary_result_file, 'Visualizaztion','VisInstanceSegmentation.png')
        dest_ism_vis_file =  "./Data/Example/SAM6D/outputs/sam6d_results/vis_ism.png"
        copy_if_exists(ism_vis_file, dest_ism_vis_file)

    # 5) Run Pose Estimation
    print("\n==> Step 5: Copy Mask")
    mask_src = os.path.join(dst, 'SAM6D', "outputs", "sam6d_results","mask_ism.png")
    mask_dst = os.path.join(dst, "FoundationPose", "masks", "000000.png")
    copy_file(mask_src, mask_dst)

    camera_src =  os.path.join(dst, 'FoundationPose', "cam_K.txt")
    # Normalize decimal separator: replace all commas with dots
    with open(camera_src, 'r', encoding='utf-8') as f:
        txt = f.read()
    txt = txt.replace(',', '.')    
    with open(camera_src, 'w', encoding='utf-8') as f:
        f.write(txt)

    if args.test_all_primary == True:
        timings['instance_segmentation_end'] = time.time()

    print("\n==> Step 6: Pose estimation in Docker")
    timings['pose_estimation_start'] = time.time()
    dst = args.example_root

    docker_cmd = [
        "docker", "exec", "-it",
        "foundationpose",
        "bash", "-lc",       # login + command string
        "cd /mnt/c/TEMP/Dennis/Bachlor/FoundationPose && python run_inference.py"
    ]

    run(docker_cmd)
    timings['pose_estimation_end'] = time.time()

    for key in list(timings.keys()):
        if key.endswith('_start'):
            step = key[:-6]  
            dur = timings[f'{step}_end'] - timings[f'{step}_start']
            timings[step] = dur
            del timings[f'{step}_start'], timings[f'{step}_end']


    # total run time
    timings['total'] = time.time() - global_start

    if args.test_all_primary == False or args.test_all_primary == "False":
        timings['total'] = timings['total'] + timings['instance_segmentation']
        timings['total'] = timings['total'] + timings['copy_templates']

    print("\n==> Step 6: Render Result")
    run(["python", "display_result.py"])

    print("\n==> Step 7: Eval Result")
    raw_models = Path(args.raw_models)
    gigapose_root = Path("../gigapose") / "gigaPose_datasets"
    raw_modelsM = Path("../SharedData/ObjectModelsM")

    gigapose_result = os.path.join(gigapose_root, "results/large_000000/predictions")
    print(f"Deleting Gigapose Result folder in {gigapose_result}")
    # Make sure it exists
    if os.path.isdir(gigapose_result):
        for name in os.listdir(gigapose_result):
            full_path = os.path.join(gigapose_result, name)
            try:
                if os.path.isfile(full_path) or os.path.islink(full_path):
                    os.unlink(full_path)              # remove files or symlinks
                elif os.path.isdir(full_path):
                    shutil.rmtree(full_path)          # remove directories and everything under them
            except Exception as e:
                print(f"Failed to delete {full_path}: {e}")
    else:
        print(f"Directory does not exist: {gigapose_result}")

    copy_model_assets(raw_modelsM, args.object_type, gigapose_root)
    copy_test_data(Path(dst), gigapose_root, "000000", Path(exp_gt),Path(exp_gt_info))
    write_scene_camera(Path(args.export_root), gigapose_root)

    run([
        "conda", "run", "--no-capture-output", "-n", "gigapose",
        "python", "-m", "src.scripts.eval_bop"
    ], cwd="../gigapose")


    print(f"Copying Results into Folder")
    scene_name = args.scene_name
    result_dst = os.path.join("..", "Result", args.object_type, args.visibility, args.lighting, args.distance, args.height, scene_name, "FoundationPose")
    img_result_dst = os.path.join(result_dst, "Images")
    vis_result_dst = os.path.join(result_dst, "Visualizaztion")
    pose_result_dst = os.path.join(result_dst, "Pose")
    score_result_dst = os.path.join(result_dst, "Scores")


    img_result_src = os.path.join(args.export_root, "MegaPose")
    copy_if_exists(img_result_src, img_result_dst, True)
    
    vis_result_src = os.path.join("mesh_visualizations", "mesh_with_corners.png")
    copy_if_exists(vis_result_src, os.path.join(vis_result_dst, "VisPoseEstimation.png"))

    result_src = os.path.join(dst, "outputs")
    copy_if_exists(os.path.join(result_src, "SAM6D", "outputs", "sam6d_results", "vis_ism.png"), os.path.join(vis_result_dst, "VisInstanceSegmentation.png"))
    
    sam_result_src = os.path.join(dst, "SAM6D", "outputs", "sam6d_results")
    pose_result_src = os.path.join(sam_result_src,  "detection_pem.json")
    copy_if_exists(pose_result_src, os.path.join(pose_result_dst, "PoseEstimation.json"))

    pose_result_src = os.path.join(sam_result_src,  "detection_ism.json")
    copy_if_exists(pose_result_src, os.path.join(pose_result_dst, "InstanceSegmentation.json"))
    
    pose_result_src = os.path.join(sam_result_src, "mask_ism.png")
    copy_if_exists(pose_result_src, os.path.join(pose_result_dst, "MaskISM.png"))

    score_result_src = os.path.join(result_src, "predictions.csv")
    copy_if_exists(score_result_src, os.path.join(score_result_dst, "predictions.csv"))

    
    score_result_src = os.path.join("../gigapose","gigaPose_datasets", "results", "large_000000", "predictions", "large-pbrreal-rgb-mmodel_lmo-test_000000")
    copy_if_exists(score_result_src, score_result_dst, True)
    
    # save to disk (e.g. under your score_result_dst/Scores)
    os.makedirs(score_result_dst, exist_ok=True)
    with open(os.path.join(score_result_dst, 'timings.json'), 'w') as f:
        json.dump(timings, f, indent=2)

    if args.test_all == True:
        run([
            "conda", "run", "--no-capture-output", "-n", "sam6d",
            "python", "pipeline.py",
            "--object_type", args.object_type,
            "--scene_name", scene_name,
            "--visibility", args.visibility,
            "--lighting", args.lighting,
            "--distance", args.distance,
            "--height", args.height,
            "--test_all_primary", "False",          
            "--primary_result_file", result_dst,
        ], cwd="../SAM-6D/SAM-6D")

        run([
            "conda", "run", "--no-capture-output", "-n", "gigapose",
            "python", "pipeline.py",
            "--object_type", args.object_type,
            "--scene_name", scene_name,
            "--visibility", args.visibility,
            "--lighting", args.lighting,
            "--distance", args.distance,
            "--height", args.height,
            "--test_all_primary", "False",          
            "--primary_result_file", result_dst,
        ], cwd="../gigapose")

        # run([
        #     "conda", "run", "--no-capture-output", "-n", "sam6d",
        #     "python", "pipeline.py",
        #     "--object_type", args.object_type,
        #     "--scene_name", scene_name,
        #     "--test_all_primary", "False",          
        #     "--primary_result_file", result_dst,
        # ], cwd="../FoundationPose")

        run([
            "conda", "run", "--no-capture-output", "-n", "megapose",
            "python", "pipeline.py",
            "--object_type", args.object_type,
            "--scene_name", scene_name,
            "--visibility", args.visibility,
            "--lighting", args.lighting,
            "--distance", args.distance,
            "--height", args.height,
            "--test_all_primary", "False",          
            "--primary_result_file", result_dst,
        ], cwd="../megapose6d")

        run([
            "conda", "run", "--no-capture-output", "-n", "ove6d",
            "python", "pipeline.py",
            "--object_type", args.object_type,
            "--scene_name", scene_name,
            "--visibility", args.visibility,
            "--lighting", args.lighting,
            "--distance", args.distance,
            "--height", args.height,
            "--test_all_primary", "False",          
            "--primary_result_file", result_dst,
        ], cwd="../OVE6D-pose")

    print("\n Pipeline complete. Results are in:", os.path.join(dst, "outputs"))

if __name__ == "__main__":
    main()
