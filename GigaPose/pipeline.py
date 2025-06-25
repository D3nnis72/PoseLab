#!
"""
python pipeline.py \
  --input_dir "../SharedData/WebsocketData/000000" \
  --bopm_output "lmo" \
  --cad_ply "./gigaPose_datasets/datasets/lmo/models/obj_000001.ply" \
  --checkpoint "sam_vit_h.pth" \
  --default_detect_dest "./gigaPose_datasets/datasets/default_detections/core19_model_based_unseen/cnos-fastsam" \
  --run_id "000000" \
  --test_dataset "lmo"

"""
import argparse
import subprocess
import shutil
import os
import sys
import time
import json


def run(cmd, cwd=None):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"Step failed: {' '.join(cmd)}", file=sys.stderr)
        sys.exit(result.returncode)

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

def timed_step(name, fn, *args, **kwargs):
    """Helper: run fn(*args, **kwargs), time it, return fn’s return and elapsed."""
    start = time.time()
    ret = fn(*args, **kwargs)
    elapsed = time.time() - start
    print(f"[{name}] took {elapsed:.2f}s")
    return ret, elapsed

def main():
    parser = argparse.ArgumentParser(
        description="Automate GigaPose segmentation and inference pipeline"
    )
    parser.add_argument("--input_dir", default="../SharedData/WebsocketData/000000",
                        help="Source folder for convert_to_bopm step")
    parser.add_argument("--bopm_output",  default="lmo",
                        help="Output name/folder for BOP-M conversion")
    parser.add_argument("--cad_ply", default="./gigaPose_datasets/datasets/lmo/models/obj_000001.ply",
                        help="Path to OBJ/PLY file for template rendering step")
    parser.add_argument("--checkpoint", default="sam_vit_h.pth",
                        help="SAM model checkpoint filename")
    parser.add_argument("--default_detect_dest", default="./gigaPose_datasets/datasets/default_detections/core19_model_based_unseen/cnos-fastsam",
                        help="Destination folder for default detections copy step")
    parser.add_argument("--run_id", default="000000",
                        help="Run ID for test inference step")
    parser.add_argument("--test_dataset", default="lmo",
                        help="Name of the test dataset (e.g. lmo)")
    parser.add_argument(
      "--reuse_templates",
      default=True,
      action="store_true",
      help="if set, skip calculating/rendering templates & just copy from raw cache"
    )
    parser.add_argument(
      "--object_type",
      default="brake",
      help="root where you keep precomputed model_info, segmentation_templates, bop_templates"
    )
    parser.add_argument(
      "--raw_cache",
      default="../SharedData/ObjectModelsM",
      help="root where you keep precomputed model_info, segmentation_templates, bop_templates"
    )
    parser.add_argument(
      "--scene_name",
      default="Default"
    )
    parser.add_argument("--test_all",     default=False,
                   help="Test all Pipelines")
    parser.add_argument("--test_all_primary", default=True)
    parser.add_argument("--primary_result_file")
    parser.add_argument("--visibility",     default="Visibility")
    parser.add_argument("--lighting",     default="Lighting")
    parser.add_argument("--distance",     default="Distance")
    parser.add_argument("--height",     default="Height")

    
    args = parser.parse_args()

    

    timings = {}


    if args.test_all_primary == False or args.test_all_primary == "False":
        timings_file = os.path.join(args.primary_result_file, 'Scores' , 'timings.json')
        if args.primary_result_file and os.path.isfile(timings_file):
            with open(timings_file, "r") as f:
              timings = json.load(f)


    global_start = time.time()


    # Step 2: Convert to BOP-M
    timings['copy_input_start'] = time.time()
    run(["python", "convert_to_bopm.py",
         "--input_dir", os.path.join(args.input_dir),
         "--output_dir", os.path.join(".", "gigaPose_datasets/datasets", args.bopm_output)])


    from PIL import Image    
    rgb_image = os.path.join(".", "gigaPose_datasets/datasets", args.bopm_output, "test", "000000","rgb", "000000.png")
    img = Image.open(rgb_image).convert("RGB")
    img.save(rgb_image)

    # Step 3.1: Render templates via BlenderProc
    if args.reuse_templates:
        print("[pipeline] --reuse_templates set, copying precomputed assets…")
        # 4a) model info
        src_info = os.path.join(
          args.raw_cache,
          args.object_type,
          "model",  
          "models_info.json"
        )
        src_model = os.path.join(
          args.raw_cache,
          args.object_type,
          "model",   
          "obj_000001.ply"
        )
        dst_info_dir = "./gigaPose_datasets/datasets/lmo/models"
        dst_eval_dir = "./gigaPose_datasets/datasets/lmo/models_eval"

        print("Model Info:")
        print(src_info)
        print(dst_info_dir)

       

        copy_if_exists(src_info, os.path.join(dst_info_dir, "models_info.json"))
        copy_if_exists(src_info, os.path.join(dst_eval_dir, "models_info.json"))

        copy_if_exists(src_model, os.path.join(dst_info_dir, "obj_000001.ply"))
        copy_if_exists(src_model, os.path.join(dst_eval_dir, "obj_000001.ply"))

        # 5a) segmentation templates

        if args.test_all_primary == True:
          src_seg = os.path.join(
            args.raw_cache,
            args.object_type,
            "segmentation_templates",   
          )
          dst_seg = "./gigaPose_datasets/datasets/lmo/"

          print("segmentation templates:")
          print(src_seg)
          print(dst_seg)

          copy_if_exists(src_seg, dst_seg, is_dir=True)

        # 5b) BOP templates for this dataset
        src_bop = os.path.join(
          args.raw_cache,
          args.object_type,
          "bop_templates",   
        )
        dst_bop = os.path.join("gigaPose_datasets", "datasets", "templates", args.bopm_output)
        print("BOP templates:")
        print(src_bop)
        print(dst_bop)
        copy_if_exists(src_bop, dst_bop, is_dir=True)

        src_obj_poses = os.path.join(
          args.raw_cache,
          args.object_type,
          "object_poses",  
        )
        dest_obj_poses = "./gigaPose_datasets/datasets/templates/lmo/object_poses"

        copy_if_exists(src_obj_poses, dest_obj_poses, is_dir=True)
    else:
        bp_out = os.path.join(".","gigaPose_datasets/datasets", args.bopm_output)
        run([
            "blenderproc", "run", "./render/render_templates_optimized.py",
            "--output_dir", bp_out,
            "--cad_path", args.cad_ply
        ])

    timings['copy_input_end'] = time.time()

    # Step 3.2: Batch inference with SAM
    ism_dir = os.path.join("Instance_Segmentation_Model")
    if args.test_all_primary == True:
      timings['instance_segmentation_start'] = time.time()
      run([
          "python", "batch_inference.py",
          "--bopm_root", os.path.join("..", "gigaPose_datasets/datasets", args.bopm_output),
          "--cad_root", os.path.join("..", "gigaPose_datasets/datasets", args.bopm_output, "models"),
          "--checkpoint", args.checkpoint
      ], cwd=ism_dir)
      timings['instance_segmentation_end'] = time.time()
    else:
        print("Copy Segmentation Result")
        ism_result_file = os.path.join(args.primary_result_file, 'Pose','InstanceSegmentation.json')
        dest_ism_result_file =  "./gigaPose_datasets/datasets/lmo/test/000000/sam6d_results/detection_ism.json"
        copy_if_exists(ism_result_file, dest_ism_result_file)

        ism_mask_file = os.path.join(args.primary_result_file, 'Pose','MaskISM.png')
        dest_ism_mask_file =  "./gigaPose_datasets/datasets/lmo/test/000000/sam6d_results/mask_ism.png"
        copy_if_exists(ism_mask_file, dest_ism_mask_file)
        
        ism_vis_file = os.path.join(args.primary_result_file, 'Visualizaztion','VisInstanceSegmentation.png')
        dest_ism_vis_file =  "./gigaPose_datasets/datasets/lmo/test/000000/sam6d_results/vis_ism.png"
        copy_if_exists(ism_vis_file, dest_ism_vis_file)

    # Step 3.3: Combine detections
    timings['pose_estimation_start'] = time.time()
    det_file = f"{args.test_dataset}_detections.json"
    run([
        "python", "combine_detection_ism.py",
        "--test_root", os.path.join("..", "gigaPose_datasets/datasets", args.bopm_output, "test"),
        "--output", os.path.join("..", "gigaPose_datasets/datasets", args.bopm_output, det_file)
    ], cwd=ism_dir)



    # Step 3.4: Copy segmentation to default_detections
    src = os.path.join(".","gigaPose_datasets/datasets", args.bopm_output, det_file)
    dst = os.path.join(args.default_detect_dest, det_file)
    print(f"Copying {src} to {dst}")
    shutil.copy(src, dst)

    # Step 4: Calculate model info
    if not args.reuse_templates:
        run(["python", "calculate_model_info.py"])

        # Step 5: Render BOP templates
        run(["python", "-m", "src.scripts.render_bop_templates"])

    # Step 6: Prepare dataset
    run(["python", "-m", "src.scripts.prepareDataset"])
   

    # Step 7: Run inference test
    run([
        "python", "test.py",
        f"test_dataset_name={args.test_dataset}",
        f"run_id={args.run_id}",
        "test_setting=localization"
    ])
    timings['pose_estimation_end'] = time.time()

    # Step 8: Visualize
    for key in list(timings.keys()):
        if key.endswith('_start'):
            step = key[:-6]  # strip "_start"
            dur = timings[f'{step}_end'] - timings[f'{step}_start']
            timings[step] = dur
            # clean up the raw start/end if you like
            del timings[f'{step}_start'], timings[f'{step}_end']


    # total run time
    timings['total'] = time.time() - global_start

    if args.test_all_primary == False or args.test_all_primary == "False":
        timings['total'] = timings['total'] + timings['instance_segmentation']
        timings['total'] = timings['total'] + timings['copy_templates']

    run(["python", "display_result.py"])


    run(["python", "-m", "src.scripts.eval_bop"])


    print(f"Copying Results into Folder")
    scene_name = args.scene_name
    result_dst = os.path.join("..", "Result", args.object_type, args.visibility, args.lighting, args.distance, args.height, scene_name, "GigaPose")
    img_result_dst = os.path.join(result_dst, "Images")
    vis_result_dst = os.path.join(result_dst, "Visualizaztion")
    pose_result_dst = os.path.join(result_dst, "Pose")
    score_result_dst = os.path.join(result_dst, "Scores")

    img_result_src = os.path.join("gigaPose_datasets", "datasets", "lmo", "test", "000000" )
    copy_if_exists(img_result_src, img_result_dst, True)
    vis_result_src = os.path.join("mesh_visualizations", "mesh_with_corners.png")
    copy_if_exists(vis_result_src, os.path.join(vis_result_dst, "VisPoseEstimation.png"))
    copy_if_exists(os.path.join(img_result_src, "sam6d_results", "vis_ism.png"), os.path.join(vis_result_dst, "VisInstanceSegmentation.png"))

    
    sam_result_src = os.path.join(img_result_src, "sam6d_results")
    pose_result_src = os.path.join(sam_result_src,  "detection_pem.json")
    copy_if_exists(pose_result_src, os.path.join(pose_result_dst, "PoseEstimation.json"))

    pose_result_src = os.path.join(sam_result_src,  "detection_ism.json")
    copy_if_exists(pose_result_src, os.path.join(pose_result_dst, "InstanceSegmentation.json"))
    
    pose_result_src = os.path.join(sam_result_src, "mask_ism.png")
    copy_if_exists(pose_result_src, os.path.join(pose_result_dst, "MaskISM.png"))

    pose_result_src = os.path.join("gigaPose_datasets", "results", "large_000000", "predictions", "large-pbrreal-rgb-mmodel_lmo-test_000000.csv")
    copy_if_exists(pose_result_src, os.path.join(pose_result_dst, "PoseEstimation.csv"))
    
    score_result_src = os.path.join("gigaPose_datasets", "results", "large_000000", "predictions", "large-pbrreal-rgb-mmodel_lmo-test_000000")
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
        ], cwd="../FoundationPose")

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

    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
