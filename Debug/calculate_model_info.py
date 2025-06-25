import os
import json
import numpy as np
from scipy.spatial.distance import pdist
from bop_toolkit_lib import inout

import numpy as np

def compute_model_info(verts):
    mn = verts.min(axis=0)
    mx = verts.max(axis=0)
    size = mx - mn
    diameter = float(np.linalg.norm(size)) 
    return {
        "diameter": diameter,
        "min_x": float(mn[0]), "min_y": float(mn[1]), "min_z": float(mn[2]),
        "size_x": float(size[0]), "size_y": float(size[1]), "size_z": float(size[2]),
    }

def compute_model_info_mm(verts):
    mn = verts.min(axis=0)
    mx = verts.max(axis=0)
    size = mx - mn
    diameter = float(np.linalg.norm(size)) 
    return {
        "diameter": diameter * 1000,
        "min_x": float(mn[0]) * 1000, "min_y": float(mn[1]) * 1000, "min_z": float(mn[2]) * 1000,
        "size_x": float(size[0]) * 1000, "size_y": float(size[1]) * 1000, "size_z": float(size[2]) * 1000,
    }


def load_vertices_from_ply(ply_path):
    raw = inout.load_ply(ply_path)
    if isinstance(raw, dict):
        verts = raw.get("pts", raw.get("vertices", None))
        if verts is None:
            raise RuntimeError(f"No vertices found in {ply_path}")
    else:
        verts = raw[0]
    return np.array(verts, dtype=np.float32)

def build_models_info(ply_dir, symmetries_dict=None):
    """
    Args:
      ply_dir: directory containing files named obj_000001.ply, obj_000005.ply, ...
      symmetries_dict: optional dict mapping int model_id -> list of 4x4 symmetry matrices
    Returns:
      info: dict of str(model_id) -> info dict
    """
    info = {}
    for fn in sorted(os.listdir(ply_dir)):
        if not fn.endswith(".ply"):
            continue

        model_id = int(fn.replace("obj_", "").replace(".ply", ""))
        ply_path = os.path.join(ply_dir, fn)
        verts = load_vertices_from_ply(ply_path)
        entry = compute_model_info(verts)
        # if you have symmetries, stick them in
        if symmetries_dict and model_id in symmetries_dict:
            entry["symmetries_discrete"] = symmetries_dict[model_id]
        info[str(model_id)] = entry
    return info

if __name__ == "__main__":
    # === USER CONFIG ===
    PLY_DIR = "../SharedData/ObjectModelsM/brake/model"        # where your .ply files live
    OUT_DIR = "../SharedData/ObjectModelsM/brake/model"      # where models_info.json will go

    os.makedirs(OUT_DIR, exist_ok=True)

    # OPTIONAL: if you already have a dict of symmetries, load it here:
    # symmetries = json.load(open("/path/to/symmetries.json", "r"))
    # otherwise just set symmetries = None
    symmetries = None

    models_info = build_models_info(PLY_DIR, symmetries)

    out_path = os.path.join(OUT_DIR, "models_info.json")
    with open(out_path, "w") as f:
        json.dump(models_info, f, indent=2)
    print(f"Saved models_info for {len(models_info)} models to {out_path}")
