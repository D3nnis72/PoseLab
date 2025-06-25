#!/usr/bin/env python3
import argparse, os, json
import numpy as np
import cv2
import trimesh
import pandas as pd
import ast  

# 12 edges between the 8 AABB corners
BOX_EDGES = [
    (0,1),(1,3),(3,2),(2,0),
    (4,5),(5,7),(7,6),(6,4),
    (0,4),(1,5),(2,6),(3,7),
]

def load_mesh(ply_path):
    mesh = trimesh.load(ply_path, process=False)
    verts = np.asarray(mesh.vertices, np.float32)  # (V,3)
    faces = np.asarray(mesh.faces,    np.int32)    # (F,3)

    mn, mx = verts.min(axis=0), verts.max(axis=0)
    corners = np.array([[x,y,z] for x in (mn[0],mx[0])
                              for y in (mn[1],mx[1])
                              for z in (mn[2],mx[2])], np.float32)
    return verts, faces, corners

def build_gt_transform(entry):
    R = np.array(entry["cam_R_m2c"], np.float32).reshape(3,3)
    t = np.array(entry["cam_t_m2c"], np.float32) / 1000.0  # mm→m
    M = np.eye(4, dtype=np.float32)
    M[:3,:3], M[:3,3] = R, t
    return M

def build_estimated_transform(R_str, t_str):
    R = np.array(list(map(float, R_str.split())), np.float32).reshape(3,3)
    t = np.array(list(map(float, t_str.split())), np.float32) / 1000.0
    M = np.eye(4, dtype=np.float32)
    M[:3,:3], M[:3,3] = R, t
    return M

def project_pts(K, pts3):
    """Project (N×3) to (N×2) float pixel coords."""
    homo = np.hstack([pts3, np.ones((pts3.shape[0],1),np.float32)])
    P = K @ np.eye(3,4, dtype=np.float32)
    proj = (P @ homo.T).T
    return proj[:,:2] / proj[:,2:3]

def render_mesh_overlay(img, K, verts, faces, M, color=(0,0,255), alpha=0.3):
    H, W = img.shape[:2]
    # transform + project all verts
    verts_cam = (M[:3,:3] @ verts.T).T + M[:3,3]
    pix = project_pts(K, verts_cam)
    pix_i = np.round(pix).astype(int)

    overlay = img.copy()
    for tri in faces:
        # skip back‐faces / off‐screen
        # if np.any(verts_cam[tri,2] <= 0): continue
        pts = pix_i[tri]
        if np.any(pts[:,0]<0) or np.any(pts[:,0]>=W) \
        or np.any(pts[:,1]<0) or np.any(pts[:,1]>=H): continue
        cv2.fillConvexPoly(overlay, pts, color)

    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

def draw_aabb_corners_sorted(img, K, corners, M):
    """
    Project the 8 AABB corners, draw their original indices in cyan,
    then draw a second label (0–7) in green sorted by cam-Z (near→far).
    """
    # 1) transform & project corners into camera space
    corners_cam = (M[:3,:3] @ corners.T).T + M[:3,3]   # (8,3)
    pix        = project_pts(K, corners_cam)           # (8,2)
    pix_i      = np.round(pix).astype(int)

    # 2) original labels (cyan)
    for idx, (x,y) in enumerate(pix_i):
        cv2.circle(img, (x,y), 4, (255,255,0), -1)
        cv2.putText(img, str(idx), (x+5,y-5),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255,255,0), 1)

    # 3) sorted‐by‐depth labels (green)
    order = np.argsort(corners_cam[:,2])  
    for rank, idx in enumerate(order):
        x,y = pix_i[idx]
        cv2.putText(img, f"{rank}", (x-25,y+25),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)

    # 4) draw the same 12 blue edges between the ORIGINAL corner indices
    for a,b in BOX_EDGES:
        cv2.line(img, tuple(pix_i[a]), tuple(pix_i[b]), (255,0,0), 2)

def main():
    p = argparse.ArgumentParser(
        description="Visualize GT mesh + AABB corners on top of RGB."
    )
    p.add_argument("--example-dir", default="./gigaPose_datasets/datasets/lmo",
                   help="Folder with MegaPose/scene_gt.json, image_rgb.png, etc.")
    p.add_argument("--model-id", type=int, default=1,
                   help="Only render this obj_id.")
    
    p.add_argument("--out-dir", default="mesh_visualizations")
    p.add_argument(
        "--model-scale", type=float, default=0.001,
        help="Scale factor to apply to mesh units → meters (e.g. 0.001 if mesh is in mm)."
    )

    args = p.parse_args()

    # load GT data
    gt = json.load(open(os.path.join(args.example_dir, "test", "000000", "scene_gt.json")))
    img = cv2.imread(os.path.join(args.example_dir, "test", "000000", "rgb", "000000.png"))
    cam = json.load(open(os.path.join(args.example_dir, "test", "000000", "camera.json")))
    ply = os.path.join(args.example_dir, "models", "obj_000001.ply")

    K   = np.array(cam["cam_K"], np.float32).reshape(3,3)

    # load mesh + AABB corners
    verts, faces, corners = load_mesh(ply)
    verts  *= args.model_scale
    corners *= args.model_scale
    print(f"Loaded mesh: {verts.shape[0]} verts, {faces.shape[0]} faces.")

    csv_path = os.path.join(
        "gigaPose_datasets", "results", "large_000000", "predictions",
        "large-pbrreal-rgb-mmodel_lmo-test_000000.csv"
    )


    # process each image / entry
    for im_str, entries in gt.items():
        print("Processing", im_str)
        if "000003" != f"000000{im_str}":
            print("Processing 2", im_str)
            for entry in entries:
                if entry["obj_id"] != args.model_id:
                    continue

                scene_id = 0             # or however your key works
                obj_id   = entry["obj_id"]
                M_gt = build_gt_transform(entry)
     
                # 1) draw filled mesh
                render_mesh_overlay(img, K, verts, faces, M_gt,
                                    color=(0,0,255), alpha=0.5)
                # 2) draw its axis‐aligned bounding‐box corners + edges
                draw_aabb_corners_sorted(img, K, corners, M_gt)

                if os.path.exists(csv_path):
                    print("esv exist")
                    df_est = pd.read_csv(csv_path)
                    # build a dict: (scene,im,obj) -> (R_str, t_str)
                    est_dict = {}
                    for row in df_est.itertuples(index=False):
                        print(row)
                        key = (row.scene_id, row.im_id, row.obj_id)
                        est_dict[key] = (row.R, row.t)
                        print(key)
                    im_id    = 0
                    key = (scene_id, im_id, obj_id)
                    if key in est_dict:
                        print("Found estimate for", key)
                        R_str, t_str = est_dict[key]
                        M_est = build_estimated_transform(R_str, t_str)
                        # render in green, more opaque
                        render_mesh_overlay(img, K, verts, faces, M_est,
                                            color=(0,255,0), alpha=0.5)
                        # Optionally draw AABB corners for estimate too:
                        draw_aabb_corners_sorted(img, K, corners, M_est)
                    else:
                        print(f"No estimate for {key}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "mesh_with_corners.png")
    cv2.imwrite(out_path, img)
    print("Wrote mesh + corner‐box overlay to", out_path)

if __name__=="__main__":
    main()
