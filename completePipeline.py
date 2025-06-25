#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
from itertools import product

def run(cmd, cwd=None):
    """
    Runs a shell command and exits on failure.
    """
    print(f"[run] {' '.join(cmd)} (cwd={cwd or os.getcwd()})")
    res = subprocess.run(cmd, cwd=cwd)
    if res.returncode != 0:
        print(f"[error] Command failed: {' '.join(cmd)}", file=sys.stderr)
        sys.exit(res.returncode)

def clear_shared_data(shared_root):
    """
    Empties the contents of shared_root and recreates it.
    """
    if os.path.isdir(shared_root):
        shutil.rmtree(shared_root)
    os.makedirs(shared_root, exist_ok=True)

def copy_input_to_shared(src_folder, shared_root):
    """
    Copies everything from src_folder into shared_root.
    """
    if not os.path.isdir(src_folder):
        raise FileNotFoundError(f"Input folder not found: {src_folder}")
    for entry in os.listdir(src_folder):
        src_path = os.path.join(src_folder, entry)
        dst_path = os.path.join(shared_root, entry)
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path)
        else:
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy(src_path, dst_path)

def main():
    parser = argparse.ArgumentParser(
        description="Batch‐run SAM‐6D pipeline and mirror results under a Result/... folder"
    )
    parser.add_argument(
        "--input_root", default="./Input",
        help="Root: Input/<Objekt>/<Sichtbarkeit>/<Beleuchtung>/<Entfernung>/<Winkel>"
    )
    parser.add_argument(
        "--result_root", default="./Result",
        help="Root für die Ausgabe: Result/<Objekt>/<Sichtbarkeit>/<Beleuchtung>/<Entfernung>/<Winkel>"
    )
    parser.add_argument(
        "--shared_root", default="./SharedData/WebsocketData/000000/0",
        help="Server‐Eingabeordner (wird jeweils geleert und neu befüllt)"
    )
    parser.add_argument(
        "--conda_env", default="sam6d",
        help="Conda‐Environment, in dem das Pipeline‐Skript läuft"
    )
    parser.add_argument(
        "--pipeline_relpath", default="./SAM-6D/SAM-6D/",
        help="Relativer Pfad zum pipeline.py (aus Arbeitsverzeichnis dieses Skripts)"
    )
    args = parser.parse_args()

    # Factors
    objects     = ["brake", "crankcase"]  
    visibilities = ["free"]
    lightings   = ["bright", "low"]
    distances   = ["75cm", "150cm"]
    heights   = ["115h", "145h"]
    angles   = ["0", "60", "120", "180", "240", "300"]
    methods  =  ["FoundationPose", "MegaPose", "GigaPose", "OVE6D", "SAM6D"]




    for obj, vis, light, dist, height, angle in product(
        objects, visibilities, lightings, distances, heights, angles
    ):
        scenario_name = f"{obj}/{vis}/{light}/{dist}/{height}/{angle}"
        print(f"\n=== Scenario: {scenario_name} ===")

        # 1) Pfad zum Eingabeordner
        src_dir = os.path.join(
            args.input_root,
            obj,
            vis,
            light,
            dist,
            height,
            angle
        )
        if not os.path.isdir(src_dir):
            print(f"[warn] No Input Folder: {src_dir}, skipping...")
            continue

        # 2) Clear Shared Data
        clear_shared_data(args.shared_root)
        copy_input_to_shared(src_dir, args.shared_root)

        # 3) Create output folder
        dst_folder = os.path.join(args.result_root, obj, vis, light, dist, height, angle)
        os.makedirs(dst_folder, exist_ok=True)

        # 4) Execute inference
        result_file = os.path.join(dst_folder, "result.json")
        cmd = [
            "conda", "run", "--no-capture-output", "-n", args.conda_env,
            "python", "pipeline.py",
            "--object_type", obj,
            "--scene_name", f"{angle}",
            "--visibility", vis,
            "--lighting", light,
            "--distance", dist,
            "--height", height,
        ]
        run(cmd, cwd=os.path.dirname(args.pipeline_relpath))

    print("\nFisnihed. Results are in:", args.result_root)

if __name__ == "__main__":
    main()
