import blenderproc as bproc
import os
import argparse
import cv2
import numpy as np
import trimesh

# argument parsing
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cad_path', required=True, help="The path of CAD model")
    parser.add_argument('--output_dir', required=True, help="The path to save CAD templates")
    parser.add_argument('--normalize', default=True, action='store_true', help="Whether to normalize CAD model or not")
    parser.add_argument('--colorize', action='store_true', help="Whether to colorize CAD model or not")
    parser.add_argument('--base_color', type=float, default=0.05, help="The base color used in CAD model")
    parser.add_argument('--material', default=False, help="Whether to colorize CAD model or not")
    return parser.parse_args()

# compute normalization scale
def get_norm_info(mesh_path):
    mesh = trimesh.load(mesh_path, force='mesh')
    pts = trimesh.sample.sample_surface(mesh, 1024)[0].astype(np.float32)
    r = max(np.linalg.norm(pts.max(axis=0)), np.linalg.norm(pts.min(axis=0)))
    return 1.0 / (2.0 * r)


def main():
    args = parse_args()
    bproc.init()

    # scale and load mesh once
    scale = get_norm_info(args.cad_path) if args.normalize else 1.0
    obj = bproc.loader.load_obj(args.cad_path)[0]
    
    obj.set_scale([scale]*3)
    obj.set_cp("category_id", 1)
    if args.colorize:
        mat = bproc.material.create('obj')
        col = [args.base_color]*3 + [0.0]
        mat.set_principled_shader_value('Base Color', col)
        obj.set_material(0, mat)

    if args.material:
        material = bproc.material.create("obj")

        # 2) check if there’s a texture file alongside your CAD model
        #    (you’ll need to adapt this to however your textures are organized)
        texture_path = os.path.splitext(args.cad_path)[0] + "_albedo.png"
        print(f"Texturepath: {texture_path}")
        if os.path.exists(texture_path):
            print("Use material")
            # create a fresh material whose Base Color is driven by that image
            material = bproc.material.create_material_from_texture(texture_path, "textured_mat")
            obj.set_material(0, material)
        else:
            # fallback to flat color
            material = bproc.material.create('obj')
            material.set_principled_shader_value('Base Color', [args.base_color]*3 + [1.0])
            obj.set_material(0, material)

    # load camera poses
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cam_file = os.path.join(base_dir, '../Instance_Segmentation_Model/utils/poses/predefined_poses/cam_poses_level0.npy')
    cam_poses = np.load(cam_file)

    # prepare output dir
    out_dir = os.path.join(args.output_dir, 'templates')
    os.makedirs(out_dir, exist_ok=True)

    # add all cameras and lights
    for pose in cam_poses:
        cam = pose.copy()
        cam[:3,1:3] *= -1
        cam[:3,-1] *= 0.001 * 2
        bproc.camera.add_camera_pose(cam)
        light = bproc.types.Light()
        light.set_type('POINT')
        light.set_location(cam[:3,-1] * 2.5)
        light.set_energy(100)

    # render batch
    bproc.renderer.set_max_amount_of_samples(50)
    data = bproc.renderer.render()
    data.update(bproc.renderer.render_nocs())

    # save each frame
    for idx in range(len(cam_poses)):
        rgb = data['colors'][idx][..., :3][..., ::-1]
        cv2.imwrite(os.path.join(out_dir, f'rgb_{idx}.png'), rgb)
        mask = (data['nocs'][idx][..., -1] * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, f'mask_{idx}.png'), mask)
        xyz = 2 * (data['nocs'][idx][..., :3] - 0.5)
        np.save(os.path.join(out_dir, f'xyz_{idx}.npy'), xyz.astype(np.float16))

if __name__ == '__main__':
    main()
