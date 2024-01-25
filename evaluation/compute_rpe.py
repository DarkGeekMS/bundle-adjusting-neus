import argparse

import numpy as np

from models.dataset import load_K_Rt_from_P


def rotation_error(pose_error):
    # compute rotation error
    a = pose_error[0, 0]
    b = pose_error[1, 1]
    c = pose_error[2, 2]
    d = 0.5 * (a + b + c - 1.0)
    rot_error = np.arccos(max(min(d, 1.0), -1.0))
    return rot_error


def translation_error(pose_error):
    # compute translation error
    dx = pose_error[0, 3]
    dy = pose_error[1, 3]
    dz = pose_error[2, 3]
    trans_error = np.sqrt(dx**2 + dy**2 + dz**2)
    return trans_error


def compute_rpe(gt, pred):
    # compute relative pose error (RPE)
    trans_errors = []
    rot_errors = []
    for i in range(len(gt) - 1):
        gt1 = gt[i]
        gt2 = gt[i + 1]
        gt_rel = np.linalg.inv(gt1) @ gt2

        pred1 = pred[i]
        pred2 = pred[i + 1]
        pred_rel = np.linalg.inv(pred1) @ pred2
        rel_err = np.linalg.inv(gt_rel) @ pred_rel

        trans_errors.append(translation_error(rel_err))
        rot_errors.append(rotation_error(rel_err))

    rpe_trans = np.mean(np.asarray(trans_errors))
    rpe_rot = np.mean(np.asarray(rot_errors))
    return rpe_trans, rpe_rot


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_path", type=str)
    parser.add_argument("--pred_path", type=str)

    args = parser.parse_args()

    gt_camera_dict = np.load(args.gt_path)
    gt_world_mats = [
        gt_camera_dict["world_mat_%d" % idx].astype(np.float32)
        for idx in range(len(gt_camera_dict) // 2)
    ]
    gt_scale_mats = [
        gt_camera_dict["scale_mat_%d" % idx].astype(np.float32)
        for idx in range(len(gt_camera_dict) // 2)
    ]
    gt_camera_poses = []
    for world_mat, scale_mat in zip(gt_world_mats, gt_scale_mats):
        P = world_mat @ scale_mat
        P = P[:3, :4]
        _, pose = load_K_Rt_from_P(None, P)
        gt_camera_poses.append(pose)

    pred_camera_dict = np.load(args.pred_path)
    pred_camera_poses = [
        pred_camera_dict["pose_mat_%d" % idx].astype(np.float32)
        for idx in range(len(pred_camera_dict) // 2)
    ]

    rpe_trans, rpe_rot = compute_rpe(gt_camera_poses, pred_camera_poses)

    print(rpe_trans, rpe_rot)
