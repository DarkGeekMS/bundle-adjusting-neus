import torch
import numpy as np
import matplotlib.pyplot as plt


def arange_pixels(resolution, batch_size=1, image_range=(-1.0, 1.0)):
    # arrange pixels for given resolution in range image_range
    # get target height and width
    h, w = resolution

    # arrange pixel location in scale resolution
    pixel_locations = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    pixel_locations = torch.stack(
        [pixel_locations[1], pixel_locations[0]],
        dim=-1).long().view(1, -1, 2).repeat(batch_size, 1, 1)

    # shift and scale points to match image_range
    pixel_scaled = pixel_locations.clone().float()
    scale = (image_range[1] - image_range[0])
    loc = (image_range[1] - image_range[0]) / 2
    pixel_scaled[:, :, 0] = scale * pixel_scaled[:, :, 0] / (w - 1) - loc
    pixel_scaled[:, :, 1] = scale * pixel_scaled[:, :, 1] / (h - 1) - loc

    return pixel_scaled


def transform_pixel_to_world(pixels, depth, camera_mat, world_mat, invert=True):
    # transform pixel positions with given depth value to world coordinates
    # invert camera matrices
    if invert:
        camera_mat = torch.inverse(camera_mat)  # 1 x 4 x 4
        world_mat = torch.inverse(world_mat)  # 1 x 4 x 4

    # transform pixels to homogeneous coordinates
    pixels = pixels.permute(0, 2, 1)  # 1 x 2 x N
    pixels = torch.cat([pixels, torch.ones_like(pixels)], dim=1)  # 1 x 4 x N

    # project pixels into camera space
    pixels_depth = pixels.clone()
    pixels_depth[:, :3] = pixels[:, :3] * depth.permute(0, 2, 1)

    # transform pixels to world space
    p_world = world_mat @ camera_mat @ pixels_depth  # 1 x 4 x N

    # transform p_world back to 3D coordinates
    p_world = p_world[:, :3].permute(0, 2, 1)  # 1 x N x 3

    return p_world


def visualize_point_cloud(points_3d, save_path):
    # export 3D point cloud visualization
    # define 3D plot
    fig = plt.figure(figsize=(18, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect("auto")
    ax.set_xlim([-2.0, 2.0])
    ax.set_ylim([-2.0, 2.0])
    ax.set_zlim([-2.0, 2.0])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # add 3D points as scatter plot
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2])

    # save visualization figure
    plt.savefig(save_path)


def comp_closest_pts_idx_with_split(pts_src, pts_des):
    # retrieve closest points between two point clouds
    pts_src_list = torch.split(pts_src, 500000, dim=1)
    idx_list = []
    for pts_src_sec in pts_src_list:
        diff = pts_src_sec[:, :, np.newaxis] - pts_des[:, np.newaxis, :]  # (3, S, 1) - (3, 1, D) -> (3, S, D)
        dist = torch.linalg.norm(diff, dim=0)  # (S, D)
        closest_idx = torch.argmin(dist, dim=1)  # (S,)
        idx_list.append(closest_idx)
    closest_idx = torch.cat(idx_list)
    return closest_idx


def comp_point_point_error(Xt, Yt):
    # compute point to point error (distance)
    closest_idx = comp_closest_pts_idx_with_split(Xt, Yt)
    pt_pt_vec = Xt - Yt[:, closest_idx]  # (3, S) - (3, S) -> (3, S)
    pt_pt_dist = torch.linalg.norm(pt_pt_vec, dim=0)
    eng = torch.mean(pt_pt_dist)
    return eng


def get_pc_loss(Xt, Yt):
    # compute point cloud loss between source and reference
    loss1 = comp_point_point_error(Xt[0].permute(1, 0), Yt[0].permute(1, 0))
    loss2 = comp_point_point_error(Yt[0].permute(1, 0), Xt[0].permute(1, 0))
    loss = loss1 + loss2
    return loss
