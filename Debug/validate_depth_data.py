import cv2
import numpy as np
import json
import matplotlib.pyplot as plt

def load_images(rgb_path, depth_path):
    # Load RGB image
    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    # Load depth image (16-bit)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    return rgb, depth

def load_camera_intrinsics(json_path, image_id):
    with open(json_path, 'r') as f:
        data = json.load(f)
    cam_data = data
    K = np.array(cam_data['cam_K']).reshape(3,3)
    depth_scale = cam_data['depth_scale']
    return K, depth_scale

def depth_to_point_cloud(depth, K, depth_scale, to_meters=True):
    """
    Convert depth image to point cloud.
    If the stored depth values are in millimeters (with depth_scale==1.0), divide by 1000 to get meters.
    """
    h, w = depth.shape
    i, j = np.indices((h, w))
    # If depth is in mm, convert to meters (if desired)
    if to_meters:
        depth = depth.astype(np.float32) / 1000.0
    else:
        depth = depth.astype(np.float32)
    # Create homogeneous pixel coordinates
    pixels = np.stack((j, i, np.ones_like(j)), axis=-1).reshape(-1, 3)
    # Invert the intrinsic matrix
    K_inv = np.linalg.inv(K)
    # Multiply each pixel by its depth
    depths = depth.reshape(-1, 1)
    points = (K_inv @ pixels.T).T * depths
    return points.reshape(h, w, 3)

def visualize_depth(depth):
    plt.figure(figsize=(8,6))
    plt.imshow(depth, cmap='plasma')
    plt.colorbar(label='Depth value')
    plt.title('Depth Map')
    plt.show()

def overlay_depth_on_rgb(rgb, depth, alpha=0.6):
    # Normalize depth for visualization
    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_PLASMA)
    overlay = cv2.addWeighted(rgb, alpha, cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB), 1 - alpha, 0)
    plt.figure(figsize=(8,6))
    plt.imshow(overlay)
    plt.title('RGB Image with Depth Overlay')
    plt.axis('off')
    plt.show()

# Example usage:
rgb_path = './Example/rgb.png'
depth_path = './Example/depth.png'
json_path = './Example/camera.json'
image_id = '000000'

# Load data
rgb, depth = load_images(rgb_path, depth_path)
K, depth_scale = load_camera_intrinsics(json_path, image_id)

print("RGB shape:", rgb.shape)
print("Depth shape:", depth.shape)
print("Camera intrinsics:\n", K)
print("Depth scale:", depth_scale)

# Visualize depth map
visualize_depth(depth)

# Create a point cloud from depth image
point_cloud = depth_to_point_cloud(depth, K, depth_scale, to_meters=True)

# Overlay depth on RGB image
overlay_depth_on_rgb(rgb, depth)

# (Optional) Further verification:
# - Check that the percentage of zero (or invalid) depth values is reasonable.
# - Reproject the point cloud into 2D (using K) and check if projected 2D points align with features in the RGB image.
invalid_ratio = np.sum(depth == 0) / depth.size
print("Invalid depth pixel ratio:", invalid_ratio)
