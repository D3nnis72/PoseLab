#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate_by_factor.py

Script that traverses the following directory structure:
    Result/
      └─ <Object>/
          └─ <Visibility>/
              └─ <Lighting>/
                  └─ <Distance>/
                      └─ <Height>/
                          └─ <Angle>/
                              └─ <Method>/
                                  └─ Scores/
                                      ├─ scores_bop19.json
                                      └─ timings.json

For each method M, a series of averaged JSON files is created.

At each level, the script filters directories against predefined lists:
`OBJECTS, VISIBILITIES, LIGHTINGS, DISTANCES, HEIGHTS, ANGLES, METHODS`.
"""

import argparse
import json
from pathlib import Path
from statistics import mean

# --------------------------------------------------------------------------------
# 1) Predefined valid names for each level
# --------------------------------------------------------------------------------

OBJECTS      = ["brake", "crankcase"]
VISIBILITIES = ["free"]
LIGHTINGS    = ["bright", "low"]     
DISTANCES    = ["75cm", "150cm"]
HEIGHTS      = ["115h", "145h"]
ANGLES       = ["0", "60", "120", "180", "240", "300"]
METHODS      = ["FoundationPose", "MegaPose", "GigaPose", "OVE6D", "SAM6D"]

# --------------------------------------------------------------------------------
# 2) Helper functions for loading, saving and aggregating JSON data
# --------------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    with open(path, 'r') as f:
        return json.load(f)

def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def aggregate_scores_dicts(dicts: list) -> dict:
    """
    Expects a list of score dictionaries with keys:
    "bop19_average_recall", "bop19_average_recall_mspd", 
    "bop19_average_recall_mssd", "bop19_average_recall_vsd", 
    "bop19_average_time_per_image"
    Returns a dict with the mean of each metric.
    """
    keys = [
        "bop19_average_recall",
        "bop19_average_recall_mspd",
        "bop19_average_recall_mssd",
        "bop19_average_recall_vsd",
        "bop19_average_time_per_image"
    ]
    agg = {}
    for k in keys:
        vals = [d.get(k, 0.0) for d in dicts]
        agg[k] = mean(vals) if vals else 0.0
    return agg

def aggregate_timings_dicts(dicts: list) -> dict:
    """
    Expects a list of timing dictionaries, e.g.:
    {"copy_input": x, "copy_templates": y, 
    "instance_segmentation": z, "pose_estimation": w, "total": t}
    Returns a dict with the union of all keys and their mean values.
    """

    all_keys = set().union(*[set(d.keys()) for d in dicts])
    agg = {}
    for k in all_keys:
        vals = [d.get(k, 0.0) for d in dicts]
        agg[k] = mean(vals) if vals else 0.0
    return agg

# --------------------------------------------------------------------------------
# 3) Utility: In einem Elternordner („base_dir“) alle validen Kinder‐Unterordner auflisten
#    Diese Funktion filtert nach predefinierten Listen (OBJECTS, VISIBILITIES, … usw.)
# --------------------------------------------------------------------------------
def valid_children(base_dir: Path, depth: int):
    """
    Returns all subfolders of base_dir whose name matches the valid list 
    for the given depth.

    depth == 1: object level       → valid: OBJECTS
    depth == 2: visibility level   → valid: VISIBILITIES
    depth == 3: lighting level     → valid: LIGHTINGS
    depth == 4: distance level     → valid: DISTANCES
    depth == 5: height level       → valid: HEIGHTS
    depth == 6: angle level        → valid: ANGLES
    depth == 7: method level       → valid: METHODS
    """

    if not base_dir.is_dir():
        return []

    if depth == 1:
        valid_list = OBJECTS
    elif depth == 2:
        valid_list = VISIBILITIES
    elif depth == 3:
        valid_list = LIGHTINGS
    elif depth == 4:
        valid_list = DISTANCES
    elif depth == 5:
        valid_list = HEIGHTS
    elif depth == 6:
        valid_list = ANGLES
    elif depth == 7:
        valid_list = METHODS
    else:
        return []

    return [c for c in base_dir.iterdir() if c.is_dir() and c.name in valid_list]

# --------------------------------------------------------------------------------
# 4) Aggregation logic for each level
# --------------------------------------------------------------------------------
def aggregate_at_level(base_dir: Path, methode_name: str):
    """
    Aggregates all score/timing JSON files from lower subfolders of a single method
    and writes *_<method>_avg.json files in base_dir.

    Details:
    • If base_dir is a distance-level folder (depth == 4), it contains height folders (depth 5):
    → Each height folder contains angle folders (depth 6), each with
        <Method>/Scores/scores_bop19.json and timings.json.
    → Load and average all JSON files from these height/angle combinations.

    • For other levels (lighting = 3, visibility = 2, object = 1):
    → Contains child folders that already have *_avg.json files.
    → Load and average these *_avg.json files.
    """
    rel_parts = base_dir.relative_to(result_root).parts
    depth = len(rel_parts)  

    if depth == 6:
        scores_in  = base_dir / methode_name / "Scores" / "scores_bop19.json"
        timings_in = base_dir / methode_name / "Scores" / "timings.json"
        if scores_in.is_file() and timings_in.is_file():
            out_scores  = base_dir / f"scores_bop19_{methode_name}_avg.json"
            out_timings = base_dir / f"timings_{methode_name}_avg.json"
            save_json(load_json(scores_in), out_scores)
            save_json(load_json(timings_in), out_timings)
            print(f"⤷ [Angle {base_dir.name}] • {methode_name}: copied raw scores/timings.")
        return


    if depth == 5:
        angle_dirs = valid_children(base_dir, 6)
        scores_list, timings_list = [], []
        for ang in angle_dirs:
            f1 = ang / methode_name / "Scores" / "scores_bop19.json"
            f2 = ang / methode_name / "Scores" / "timings.json"
            if f1.is_file() and f2.is_file():
                scores_list.append(load_json(f1))
                timings_list.append(load_json(f2))
        if scores_list:
            agg_s = aggregate_scores_dicts(scores_list)
            agg_t = aggregate_timings_dicts(timings_list)
            save_json(agg_s, base_dir / f"scores_bop19_{methode_name}_avg.json")
            save_json(agg_t, base_dir / f"timings_{methode_name}_avg.json")
            print(f"⤷ [Height {base_dir.name}] • {methode_name}: aggregated {len(scores_list)} angles.")
        return
    

    if depth == 4:
        height_dirs = valid_children(base_dir, 5)
        all_scores = []
        all_timings = []
        for height_dir in height_dirs:
            angle_dirs = valid_children(height_dir, 6)
            for angle_dir in angle_dirs:
                scores_path = angle_dir / methode_name / "Scores" / "scores_bop19.json"
                timings_path= angle_dir / methode_name / "Scores" / "timings.json"
                if scores_path.is_file() and timings_path.is_file():
                    all_scores.append(load_json(scores_path))
                    all_timings.append(load_json(timings_path))

        if all_scores and all_timings:
            agg_scores = aggregate_scores_dicts(all_scores)
            agg_timings= aggregate_timings_dicts(all_timings)
            out_scores = base_dir / f"scores_bop19_{methode_name}_avg.json"
            out_timings= base_dir / f"timings_{methode_name}_avg.json"
            save_json(agg_scores, out_scores)
            save_json(agg_timings, out_timings)
            print(f"⤷ [Distance: {base_dir.name}] • Methode {methode_name}: aus {len(all_scores)} Height×Angle‐Paaren gemittelt.")


    elif depth < 4:
        child_dirs = valid_children(base_dir, depth+1)
        scores_list = []
        timings_list = []
        for d in child_dirs:
            s_path = d / f"scores_bop19_{methode_name}_avg.json"
            t_path = d / f"timings_{methode_name}_avg.json"
            if s_path.is_file() and t_path.is_file():
                scores_list.append(load_json(s_path))
                timings_list.append(load_json(t_path))

        if scores_list and timings_list:
            agg_scores = aggregate_scores_dicts(scores_list)
            agg_timings= aggregate_timings_dicts(timings_list)
            out_scores = base_dir / f"scores_bop19_{methode_name}_avg.json"
            out_timings= base_dir / f"timings_{methode_name}_avg.json"
            save_json(agg_scores, out_scores)
            save_json(agg_timings, out_timings)
            print(f"⤷ [{base_dir.name}] • Methode {methode_name}: aus {len(scores_list)} Kind-Ordnern gemittelt.")

# --------------------------------------------------------------------------------
# 5) Main function that traverses all levels bottom-up
# --------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Aggregiert pro Methode die JSON‐Scores/Timings über die Faktoren "
                    "(Object, Visibility, Lighting, Distance, Height, Angle)."
    )
    parser.add_argument(
        "--result_root",
        default="./Result",
        help="Wurzelordner: ./Result/<Object>/<Visibility>/<Lighting>/<Distance>/<Height>/<Angle>/<Methode>/Scores/…"
    )
    args = parser.parse_args()

    global result_root
    result_root = Path(args.result_root)
    if not result_root.is_dir():
        print(f"Pfad existiert nicht: {result_root}")
        return

    # Determine all methods based on any available angle folder
    first_methods = None
    for obj_dir in valid_children(result_root, 1):
        for vis_dir in valid_children(obj_dir, 2):
            for light_dir in valid_children(vis_dir, 3):
                for dist_dir in valid_children(light_dir, 4):
                    for height_dir in valid_children(dist_dir, 5):
                        for angle_dir in valid_children(height_dir, 6):
                            print(angle_dir)
                            methods_here = [
                                m.name for m in angle_dir.iterdir()
                                if m.name in METHODS and
                                   (m / "Scores" / "scores_bop19.json").is_file()
                            ]
                            if methods_here:
                                first_methods = methods_here
                                break
                        if first_methods:
                            break
                    if first_methods:
                        break
                if first_methods:
                    break
            if first_methods:
                break
        if first_methods:
            break

    if not first_methods:
        print("Keine Methoden-Ordner mit scores_bop19.json gefunden. Prüfe die Struktur.")
        return

    # Bottom-up aggregation: from depth 6 (angle) up to depth 1 (object)
    for depth in [6, 5, 4, 3, 2, 1]:
        # collect all base_dirs at this depth
        if depth == 6:
            parent_dirs = [
                a 
                for obj in valid_children(result_root, 1)
                for vis in valid_children(obj, 2)
                for lig in valid_children(vis, 3)
                for dist in valid_children(lig, 4)
                for ht in valid_children(dist, 5)
                for a in valid_children(ht, 6)
            ]
        elif depth == 5:
            parent_dirs = [
                ht
                for obj in valid_children(result_root, 1)
                for vis in valid_children(obj, 2)
                for lig in valid_children(vis, 3)
                for dist in valid_children(lig, 4)
                for ht in valid_children(dist, 5)
            ]
        elif depth == 4:
            parent_dirs = [
                dist
                for obj in valid_children(result_root, 1)
                for vis in valid_children(obj, 2)
                for lig in valid_children(vis, 3)
                for dist in valid_children(lig, 4)
            ]
        elif depth == 3:
            parent_dirs = [
                lig
                for obj in valid_children(result_root, 1)
                for vis in valid_children(obj, 2)
                for lig in valid_children(vis, 3)
            ]
        elif depth == 2:
            parent_dirs = [
                vis
                for obj in valid_children(result_root, 1)
                for vis in valid_children(obj, 2)
            ]
        else:  # depth == 1
            parent_dirs = valid_children(result_root, 1)

        # now aggregate in each of those directories for every method
        for base_dir in parent_dirs:
            for methode in first_methods:
                aggregate_at_level(base_dir, methode)

if __name__ == "__main__":
    main()
