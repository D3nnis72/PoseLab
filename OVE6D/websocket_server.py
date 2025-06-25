# server.py
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
import csv

HOST = "0.0.0.0"
PORT = 8765
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


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    M = R.astype(np.float64)
    t = M[0,0] + M[1,1] + M[2,2]
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = ( M[2,1] - M[1,2] ) * s
        y = ( M[0,2] - M[2,0] ) * s
        z = ( M[1,0] - M[0,1] ) * s
    else:
        if M[0,0] > M[1,1] and M[0,0] > M[2,2]:
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
    # 1) launch your existing pipeline
    cmd = [
        sys.executable, PIPELINE_SCRIPT,
        "--object_type", object_type,
        "--scene_name", scene_name,
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[server] Pipeline failed: {e}", file=sys.stderr)
        return {"position":[0,0,0], "rotation":[0,0,0,1]}

    # 2) load OVE6D result JSON
    print(f"[server] Searching Result for image {image_id}")
    csv_path = os.path.join(
        ".", "Dataspace", "lm","outputs", "predictions.csv"
    )
    if not os.path.isfile(csv_path):
        print(f"[server] Missing CSV: {print}", file=sys.stderr)
        return {"position":[0,0,0], "rotation":[0,0,0,0,0,0,0,0,0]}

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        try:
            row = next(reader)
        except StopIteration:
            print("[server] CSV empty", file=sys.stderr)
            return {"position":[0,0,0], "rotation":[0,0,0,1]}

     # 4) parse R and t
    R_vals = list(map(float, row["R"].split()))
    t_vals = list(map(float, row["t"].split()))

    # reshape R into 3×3
    R = np.array(R_vals, dtype=np.float32).tolist()
    position = (np.array(t_vals, dtype=np.float32)).tolist()

    print(f"[server] Returning position={position}, rotation={R}")
    return {"position": position, "rotation": R}


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

def save_megapose_raw(folder, rgb_bytes, depth_bytes, camK):
    root = os.path.join(folder, "MegaPose")
    os.makedirs(os.path.join(root, "inputs"), exist_ok=True)
    os.makedirs(os.path.join(root, "meshes"), exist_ok=True)
    with open(os.path.join(root, "inputs", "image_rgb.png"),   "wb") as f: f.write(rgb_bytes)
    with open(os.path.join(root, "inputs", "image_depth.png"), "wb") as f: f.write(depth_bytes)
    with open(os.path.join(root, "image_rgb.png"),   "wb") as f: f.write(rgb_bytes)
    with open(os.path.join(root, "image_depth.png"), "wb") as f: f.write(depth_bytes)
    h = camK  # not used here, but resolution read by Unity side
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
            # 1) header
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

            # raw saves
            save_sam6d_raw(scene_folder, rgb_bytes, depth_bytes, camK)
            save_foundation_raw(scene_folder, rgb_bytes, depth_bytes, camK, img_id, object_type)
            save_megapose_raw(scene_folder, rgb_bytes, depth_bytes, camK)

            scene_gt_json     = hdr.get("sceneGtJson")
            scene_gt_info_json= hdr.get("sceneGtInfoJson")
            save_gt(scene_folder, scene_gt_json, scene_gt_info_json)

            scene_name = hdr.get("sceneName") 
            # estimate + reply
            pose = await estimate_pose(img_root, img_id, object_type, scene_name)
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
