import asyncio
import json
import argparse
import os
import shutil
import signal
import socket
import sys
import subprocess

import numpy as np
import cv2
import websockets
import netifaces

HOST = "0.0.0.0"
PORT = 8761
EXPORT_ROOT = "../SharedData/WebsocketData"
PIPELINE_SCRIPT = os.path.join("pipeline.py")

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except:
        return "127.0.0.1"
    finally:
        s.close()

def get_tailscale_ip():
    # look for any interface with "tailscale" in its name
    print("get tailscale")
    for iface in netifaces.interfaces():
        print(iface)
        if "tailscale" in iface.lower():
            addrs = netifaces.ifaddresses(iface).get(netifaces.AF_INET, [])
            if addrs:
                return addrs[0]["addr"]
    # if that fails, you can still try the CLI as a last resort
    try:
        out = subprocess.check_output(["tailscale","ip","-4"], stderr=subprocess.DEVNULL)
        return out.decode().split()[0]
    except Exception:
        return None

"""
def get_tailscale_ip():
    try:
        print("Test")
        addrs = netifaces.ifaddresses("tailscale0")[netifaces.AF_INET]
        print("get tailscale ip")
        print(addrs)
        return addrs[0]["addr"]
    except Exception:
        return None
"""

def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix to a (x,y,z,w) quaternion.
    Reference: https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    """
    # Make sure it's float64 for stability
    M = R.astype(np.float64)
    trace = M[0,0] + M[1,1] + M[2,2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = ( M[2,1] - M[1,2] ) * s
        y = ( M[0,2] - M[2,0] ) * s
        z = ( M[1,0] - M[0,1] ) * s
    else:
        # Find largest diagonal element
        if (M[0,0] > M[1,1]) and (M[0,0] > M[2,2]):
            s = 2.0 * np.sqrt(1.0 + M[0,0] - M[1,1] - M[2,2])
            w = (M[2,1] - M[1,2]) / s
            x = 0.25 * s
            y = (M[0,1] + M[1,0]) / s
            z = (M[0,2] + M[2,0]) / s
        elif M[1,1] > M[2,2]:
            s = 2.0 * np.sqrt(1.0 + M[1,1] - M[0,0] - M[2,2])
            w = (M[0,2] - M[2,0]) / s
            x = (M[0,1] + M[1,0]) / s
            y = 0.25 * s
            z = (M[1,2] + M[2,1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + M[2,2] - M[0,0] - M[1,1])
            w = (M[1,0] - M[0,1]) / s
            x = (M[0,2] + M[2,0]) / s
            y = (M[1,2] + M[2,1]) / s
            z = 0.25 * s
    return np.array([x, y, z, w], dtype=np.float32)

async def estimate_pose(img_folder, image_id, object_type, scene_name):
    """
    Run the full GigaPose pipeline on the saved data folder, then load/return the final pose
    from the .npz output (choosing the hypothesis with highest score).
    """
    # 1) run your existing pipeline
    input_dir = os.path.join(".", img_folder)
    cmd = [
        sys.executable, PIPELINE_SCRIPT,
        "--input_dir",       input_dir,
        "--bopm_output",     "lmo",
        "--cad_ply",         "./gigaPose_datasets/datasets/lmo/models/obj_000001.ply",
        "--checkpoint",      "sam_vit_h.pth",
        "--default_detect_dest",
        "./gigaPose_datasets/datasets/default_detections/core19_model_based_unseen/cnos-fastsam",
        "--run_id",          image_id,
        "--test_dataset",    "lmo",
        "--reuse_templates",
        "--object_type", object_type,
        "--raw_cache", "../SharedData/ObjectModelsM",
        "--scene_name", scene_name,
    ]
    try:
        print(f"[server] Running pipeline for image {image_id}...")
        subprocess.run(cmd, check=True)
        print(f"[server] Completed pipeline for image {image_id}")

    except subprocess.CalledProcessError as e:
        print(f"[server] Pipeline failed: {e}", file=sys.stderr)
        return {"position": [0,0,0], "rotation": [0,0,0,1]}
    except Exception as e:
        print(f"[server] Error sending pose: {e}", file=sys.stderr)
        # close the socket with an error code
        await ws.close(code=1011, reason="Internal Server Error")
        return


    # 2) load the .npz results
    print(f"[server] Searching Result for image {image_id}")
    npz_path = os.path.join(
        "gigaPose_datasets", "results", "large_" +str(image_id), "predictions", "0.npz"
    )
    
    if not os.path.isfile(npz_path):
        print(f"[server] WARNING: result file not found: {npz_path}", file=sys.stderr)
        return {"position": [0,0,0], "rotation": [0,0,0,1]}

    data = np.load(npz_path)
    poses  = data["poses"][0]   # shape (N,4,4)
    scores = data["scores"][0]  # shape (N,)

    # 3) pick the best hypothesis
    best_idx = int(np.argmax(scores))
    best_mat = poses[best_idx]  # 4x4

    # 4) split to translation + quaternion
    t = best_mat[:3, 3].tolist()
    R = best_mat[:3, :3]

    position_list = t                    
    rotation_list = R.flatten().tolist()   

    print(f"[server] Returning Result for image {image_id}")
    return {
        "position": position_list,
        "rotation": rotation_list
    }

def save_sam6d_raw(folder, rgb_bytes, depth_bytes, camK):
    root = os.path.join(folder, "SAM6D")
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "rgb.png"), "wb")   as f: f.write(rgb_bytes)
    with open(os.path.join(root, "depth.png"), "wb") as f: f.write(depth_bytes)
    cam_data = {"cam_K": camK.flatten().tolist(), "depth_scale": 1.0}
    with open(os.path.join(root, "camera.json"), "w") as f:
        json.dump(cam_data, f, indent=2)

def save_foundation_raw(folder, rgb_bytes, depth_bytes, camK, image_id, object_type):
    root = os.path.join(folder, "FoundationPose")
    for sub in ("rgb","depth","masks","mesh"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    fn = f"{image_id}.png"
    with open(os.path.join(root, "rgb", fn),   "wb") as f: f.write(rgb_bytes)
    with open(os.path.join(root, "depth", fn), "wb") as f: f.write(depth_bytes)
    # cam_K.txt
    lines = []
    for row in camK.reshape(3,3):
        lines.append(" ".join(f"{v:.18E}" for v in row))
    with open(os.path.join(root, "cam_K.txt"), "w") as f:
        f.write("\n".join(lines))

    if object_type:
        # where your models live; adjust as needed
        MODEL_SOURCE_ROOT = "../SharedData/ObjectModels"
        src_mesh = os.path.join(MODEL_SOURCE_ROOT, object_type, "model",f"obj_000001.ply")
        dst_mesh_dir = os.path.join(root, "mesh")
        if os.path.isfile(src_mesh):
            shutil.copy(src_mesh, dst_mesh_dir)
            print(f"[server] copied mesh for '{object_type}'")
        else:
            print(f"[server] WARNING: mesh for '{object_type}' not found at {src_mesh}")
    else:
        print("[server] WARNING: no objectType in header; skipping mesh copy")

def save_megapose_raw(folder, rgb_bytes, depth_bytes, camK):
    root = os.path.join(folder, "MegaPose")
    os.makedirs(os.path.join(root, "inputs"), exist_ok=True)
    os.makedirs(os.path.join(root, "meshes"), exist_ok=True)
    with open(os.path.join(root, "inputs", "image_rgb.png"),   "wb") as f: f.write(rgb_bytes)
    with open(os.path.join(root, "inputs", "image_depth.png"), "wb") as f: f.write(depth_bytes)
    with open(os.path.join(root, "image_rgb.png"),   "wb") as f: f.write(rgb_bytes)
    with open(os.path.join(root, "image_depth.png"), "wb") as f: f.write(depth_bytes)    

    cam_json = {
        "K": camK.reshape(3,3).tolist(),
        "resolution": [targetHeight, targetWidth]
    }
    with open(os.path.join(root, "camera_data.json"), "w") as f:
        json.dump(cam_json, f, indent=2)

def save_gt(folder, scene_gt_json, scene_gt_info_json):
    root = os.path.join(folder, "GT")
    os.makedirs(os.path.join(root), exist_ok=True)
    
    if scene_gt_json:
        with open(os.path.join(root, "scene_gt.json"), "w") as f:
            f.write(scene_gt_json)
    else:
        print("[server] WARNING: no sceneGtJson in header")

    if scene_gt_info_json:
        with open(os.path.join(root, "scene_gt_info.json"), "w") as f:
            f.write(scene_gt_info_json)
    else:
        print("[server] WARNING: no sceneGtInfoJson in header")

async def handler(ws):
    print(f"[server] Client connected: {ws.remote_address}")
    try:
        while True:
            hdr_msg = await ws.recv()
            if isinstance(hdr_msg, bytes):
                print("[server] expected header text, got bytes")
                return
            hdr = json.loads(hdr_msg)
            camK = np.array(hdr["camK"], np.float32).reshape((3,3))
            
            img_id = "000000"

            # 2) RGB bytes
            rgb_bytes = await ws.recv()
            # 3) depth bytes
            depth_bytes = await ws.recv()

            # make folder
            img_root = os.path.join(EXPORT_ROOT, img_id)
            os.makedirs(img_root, exist_ok=True)

            scene_folder = os.path.join(img_root, "0")
            os.makedirs(scene_folder, exist_ok=True)

            object_type = hdr.get("objectType")
            sceneName = hdr.get("sceneName")



            # raw saves
            save_sam6d_raw(scene_folder, rgb_bytes, depth_bytes, camK)
            save_foundation_raw(scene_folder, rgb_bytes, depth_bytes, camK, img_id, object_type)
            save_megapose_raw(scene_folder, rgb_bytes, depth_bytes, camK)

            scene_gt_json     = hdr.get("sceneGtJson")
            scene_gt_info_json= hdr.get("sceneGtInfoJson")
            save_gt(scene_folder, scene_gt_json, scene_gt_info_json)

            # estimate + reply
            pose = await estimate_pose(img_root, img_id, object_type, sceneName)
            await ws.send(json.dumps(pose))

    except websockets.ConnectionClosed:
        print("[server] Client disconnected")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()

    if args.clear and os.path.isdir(EXPORT_ROOT):
        shutil.rmtree(EXPORT_ROOT)
    os.makedirs(EXPORT_ROOT, exist_ok=True)


    # discover Tailscale address (or fall back)
    ts_ip = get_tailscale_ip() or get_local_ip()
    print(f"[server] starting on ws://{ts_ip}:{PORT}")

    server = await websockets.serve(
        handler,
        HOST, 
        PORT,
        ping_interval=5,
         ping_timeout=900,)

    loop = asyncio.get_running_loop()
    shutdown = loop.create_future()

    # signal handlers will set the future, not raise
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown.set_result, None)

    print("[server] ready; press Ctrl+C to exit")
    await shutdown

    print("\n[server] shutting down…")
    server.close()
    await server.wait_closed()
    print("[server] closed")


if __name__ == "__main__":
    TARGET = (640, 480)
    targetWidth, targetHeight = TARGET
    asyncio.run(main())
