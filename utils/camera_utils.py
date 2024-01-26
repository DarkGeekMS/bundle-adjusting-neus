from typing import Tuple, Union

import cv2
import numpy as np
import torch


# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


def convert3x4_4x4(input):
    """
    :param input:  (N, 3, 4) or (3, 4) torch or np
    :return:       (N, 4, 4) or (4, 4) torch or np
    """
    if torch.is_tensor(input):
        if len(input.shape) == 3:
            output = torch.cat(
                [input, torch.zeros_like(input[:, 0:1])], dim=1
            )  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = torch.cat(
                [
                    input,
                    torch.tensor(
                        [[0, 0, 0, 1]], dtype=input.dtype, device=input.device
                    ),
                ],
                dim=0,
            )  # (4, 4)
    else:
        if len(input.shape) == 3:
            output = np.concatenate(
                [input, np.zeros_like(input[:, 0:1])], axis=1
            )  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = np.concatenate(
                [input, np.array([[0, 0, 0, 1]], dtype=input.dtype)], axis=0
            )  # (4, 4)
            output[3, 3] = 1.0
    return output


def vec2skew(v):
    """
    :param v:  (3, ) torch tensor
    :return:   (3, 3)
    """
    zero = torch.zeros(1, dtype=torch.float32, device=v.device)
    skew_v0 = torch.cat([zero, -v[2:3], v[1:2]])  # (3, 1)
    skew_v1 = torch.cat([v[2:3], zero, -v[0:1]])
    skew_v2 = torch.cat([-v[1:2], v[0:1], zero])
    skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=0)  # (3, 3)
    return skew_v  # (3, 3)


def Exp(r):
    """
    so(3) vector to SO(3) matrix
    :param r: (3, ) axis-angle, torch tensor
    :return:  (3, 3).
    """
    skew_r = vec2skew(r)  # (3, 3)
    norm_r = r.norm() + 1e-15
    eye = torch.eye(3, dtype=torch.float32, device=r.device)
    R = (
        eye
        + (torch.sin(norm_r) / norm_r) * skew_r
        + ((1 - torch.cos(norm_r)) / norm_r**2) * (skew_r @ skew_r)
    )
    return R


def make_c2w(r, t):
    """
    :param r:  (3, ) axis-angle             torch tensor
    :param t:  (3, ) translation vector     torch tensor
    :return:   (4, 4)
    """
    R = Exp(r)  # (3, 3)
    c2w = torch.cat([R, t.unsqueeze(1)], dim=1)  # (3, 4)
    c2w = convert3x4_4x4(c2w)  # (4, 4)
    return c2w


def make_calib(fx, fy, cx, cy):
    """
    :param fx:  (1, ) focal length on x     torch tensor
    :param fy:  (1, ) focal length on y     torch tensor
    :param cx:  (1, ) camera center x       torch tensor
    :param cy:  (1, ) camera center y       torch tensor
    :return:   (4, 4)
    """
    zero = torch.tensor(0, dtype=torch.float32, device=fx.device)
    one = torch.tensor(1, dtype=torch.float32, device=fx.device)
    fx_v = torch.stack([fx, zero])
    fy_v = torch.stack([zero, fy])
    f_sub_mat = torch.stack([fx_v, fy_v], dim=1)
    k_matrix = torch.cat(
        [
            torch.cat(
                [f_sub_mat, torch.reshape(torch.stack([cx, cy]), shape=(2, 1))], dim=1
            ),
            torch.reshape(torch.stack([zero, zero, one]), shape=(1, 3)),
        ],
        dim=0,
    )
    k_matrix = torch.cat(
        [k_matrix, torch.reshape(torch.stack([zero, zero, zero]), shape=(3, 1))], dim=1
    )
    k_matrix = torch.cat(
        [k_matrix, torch.reshape(torch.stack([zero, zero, zero, one]), shape=(1, 4))],
        dim=0,
    )
    return k_matrix


def scale_camera(cam: Union[np.ndarray, torch.Tensor], scale: Union[Tuple, float] = 1):
    """resize input in order to produce sampled depth map"""
    if type(scale) != tuple:
        scale = (scale, scale)
    if type(cam) == np.ndarray:
        new_cam = np.copy(cam)
        # focal:
        new_cam[1, 0, 0] = cam[1, 0, 0] * scale[0]
        new_cam[1, 1, 1] = cam[1, 1, 1] * scale[1]
        # principle point:
        new_cam[1, 0, 2] = cam[1, 0, 2] * scale[0]
        new_cam[1, 1, 2] = cam[1, 1, 2] * scale[1]
    elif type(cam) == torch.Tensor:
        new_cam = cam.clone()
        # focal:
        new_cam[..., 1, 0, 0] = cam[..., 1, 0, 0] * scale[0]
        new_cam[..., 1, 1, 1] = cam[..., 1, 1, 1] * scale[1]
        # principle point:
        new_cam[..., 1, 0, 2] = cam[..., 1, 0, 2] * scale[0]
        new_cam[..., 1, 1, 2] = cam[..., 1, 1, 2] * scale[1]
    else:
        raise TypeError
    return new_cam
