import argparse
import os

import bpy
import numpy as np
from scipy.spatial.transform import Rotation

from utils.camera_utils import load_K_Rt_from_P


def read_camera(camera_path, is_colmap):
    # load camera
    if is_colmap:
        camera_params = np.load(camera_path)
        camera_poses = [
            camera_params[cam_id][:, :4] for cam_id in range(len(camera_params))
        ]
    else:
        camera_dict = np.load(camera_path)
        try:
            camera_poses = [
                camera_dict["pose_mat_%d" % idx].astype(np.float32)
                for idx in range(len(camera_dict) // 2)
            ]
        except Exception:
            world_mats = [
                camera_dict["world_mat_%d" % idx].astype(np.float32)
                for idx in range(len(camera_dict) // 2)
            ]
            scale_mats = [
                camera_dict["scale_mat_%d" % idx].astype(np.float32)
                for idx in range(len(camera_dict) // 2)
            ]
            camera_poses = []
            for world_mat, scale_mat in zip(world_mats, scale_mats):
                P = world_mat @ scale_mat
                P = P[:3, :4]
                _, pose = load_K_Rt_from_P(None, P)
                camera_poses.append(pose)

    # extract rotations and translations
    trans_list = []
    rot_list = []
    for camera_pose in camera_poses:
        trans_list.append(camera_pose[:3, 3].tolist())
        r = Rotation.from_matrix(camera_pose[:3, :3])
        angles = r.as_euler("xyz", degrees=False)
        rot_list.append(angles.tolist())

    return trans_list, rot_list


def render_mesh_from_camera(
    mesh_path, camera_path, camera_lens, light_position, output_path, is_colmap
):
    # clear blender scene
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    for this_obj in bpy.data.objects:
        if this_obj.type == "MESH":
            this_obj.select_set(True)
            bpy.ops.object.delete(use_global=False, confirm=False)

    # import and load mesh object
    _ = bpy.ops.import_mesh.ply(filepath=mesh_path)

    for this_obj in bpy.data.objects:
        if this_obj.type == "MESH":
            this_obj.select_set(True)
            bpy.context.view_layer.objects.active = this_obj
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.split_normals()

    bpy.ops.object.mode_set(mode="OBJECT")

    # read camera views
    trans_list, rot_list = read_camera(camera_path, is_colmap)

    # loop over all camera views
    for cam_id, (trans, rot) in enumerate(zip(trans_list, rot_list)):
        # setup camera view
        cam = bpy.data.objects["Camera"]
        cam.data.lens = camera_lens
        cam.location.x = trans[0]
        cam.location.y = trans[1]
        cam.location.z = trans[2]
        cam.rotation_euler[0] = rot[0]
        cam.rotation_euler[1] = rot[1]
        cam.rotation_euler[2] = rot[2]
        if not is_colmap:
            cam.scale = (-1.0, -1.0, -1.0)

        # setup scene light
        light = bpy.data.objects["Light"]
        light.location.z = light_position

        # render mesh from camera view
        bpy.context.scene.render.image_settings.color_mode = "RGBA"
        bpy.context.scene.render.film_transparent = True
        bpy.context.scene.render.filepath = f"{output_path}/{cam_id}.png"
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_path", type=str)
    parser.add_argument("--camera_path", type=str)
    parser.add_argument("--camera_lens", type=float, default=30.0)
    parser.add_argument("--light_position", type=float, default=-10.0)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--is_colmap", default=False, action="store_true")

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    render_mesh_from_camera(
        args.mesh_path,
        args.camera_path,
        args.camera_lens,
        args.light_position,
        args.output_path,
        args.is_colmap,
    )
