import argparse
import os
from pathlib import Path
from run_inference_custom import run_inference

def batch_bopm_inference(
    segmentor_model: str,
    bopm_root: str,
    cad_root: str,
    checkpoint: str,
    stability_score_thresh: float = 0.97,
):
    """
    For every scene folder under bopm_root/test, find its rgb, depth, cam and
    call run_inference, dumping detections.json into that scene folder.
    """
    bopm_root = Path(bopm_root)
    template_path = os.path.join(bopm_root, 'templates')
    test_root = bopm_root / "test"
    cad_root = Path(cad_root)
        
    for scene_dir in sorted(test_root.iterdir()):
        if not scene_dir.is_dir(): 
            continue
 
        scene_id = int(scene_dir.name)

        rgb_dir   = scene_dir / "rgb"
        depth_dir = scene_dir / "depth"
        cam_file  = scene_dir / "camera.json"  
        out_dir   = scene_dir
        output_subdir = scene_dir / "sam6d_results"
        output_subdir.mkdir(exist_ok=True, parents=True)

        for rgb_path in sorted(rgb_dir.glob("*.png")):
            im_id = int(rgb_path.stem)  

            run_inference(
                segmentor_model=segmentor_model,
                output_dir=str(out_dir),
                cad_path=str(cad_root / "obj_000001.ply"),
                rgb_path=str(rgb_path),
                depth_path=str(depth_dir / f"{im_id:06d}.png"),
                cam_path=str(cam_file),
                stability_score_thresh=stability_score_thresh,
                template_path=str(template_path),
                scene_id=scene_id,       
                 image_id=im_id           
            )

  

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--segmentor_model", default="sam")
    p.add_argument("--bopm_root",     required=True,
                   help="Root of your BOP-M dataset (contains test/)")
    p.add_argument("--cad_root",      required=True,
                   help="Folder with your models, e.g. models/obj_000001.ply, obj_000002.ply, …")
    p.add_argument("--checkpoint",    required=True,
                   help="Path to SAM2 .pth checkpoint")
    p.add_argument("--stability_thresh", type=float, default=0.97)
    args = p.parse_args()

    batch_bopm_inference(
        args.segmentor_model,
        args.bopm_root,
        args.cad_root,
        args.checkpoint,
        stability_score_thresh=args.stability_thresh,
    )
