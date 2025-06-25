import asyncio
import json
import numpy as np
import websockets
import argparse
import os

HOST = '100.69.178.95'
PORT = 8765

async def send_from_folder(folder):
    uri = f"ws://{HOST}:{PORT}"
    # Load camera intrinsics (replace commas with dots for float conversion)
    cam_file = os.path.join(folder, 'cam_K.txt')
    with open(cam_file, 'r') as f:
        lines = f.read().splitlines()
    cam_data = []
    for line in lines:
        # unify decimal separator
        clean_line = line.replace(',', '.')
        cam_data.append([float(val) for val in clean_line.split()])
    camK = np.array(cam_data, dtype=np.float32)

    image_id = os.path.basename(folder.rstrip(os.sep))

    # Read RGB PNG bytes
    rgb_path = os.path.join(folder, 'rgb', f'000000.png')
    with open(rgb_path, 'rb') as f:
        rgb_bytes = f.read()

    # Read Depth PNG bytes
    depth_path = os.path.join(folder, 'depth', f'000000.png')
    with open(depth_path, 'rb') as f:
        depth_bytes = f.read()

    async with websockets.connect(uri, ping_interval=60,    # reply to server pings every 60 s
    ping_timeout=900     # wait up to 15 min for server pings
)  as ws:
        # 1) send header
        header = {
            'camK': camK.flatten().tolist(),
            'imageID': "0",
            "objectType": "crankcase"
        }
        await ws.send(json.dumps(header))

        # 2) send rgb bytes
        await ws.send(rgb_bytes)

        # 3) send depth bytes
        await ws.send(depth_bytes)

        # 4) receive pose
        pose_msg = await ws.recv()
        pose = json.loads(pose_msg)
        print('Received pose for', image_id, ':', pose)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test WebSocket server with folder data')
    parser.add_argument('folder', help='Path to FoundationPose folder')
    args = parser.parse_args()
    asyncio.run(send_from_folder(args.folder))
