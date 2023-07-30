import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.transform import Rotation as RotLib


def SO3_to_quat(R):
    """
    :param R:  (N, 3, 3) or (3, 3) np
    :return:   (N, 4, ) or (4, ) np
    """
    x = RotLib.from_matrix(R)
    quat = x.as_quat()
    return quat


def quat_to_SO3(quat):
    """
    :param quat:    (N, 4, ) or (4, ) np
    :return:        (N, 3, 3) or (3, 3) np
    """
    x = RotLib.from_quat(quat)
    R = x.as_matrix()
    return R


def convert3x4_4x4(input):
    """
    :param input:  (N, 3, 4) or (3, 4) torch or np
    :return:       (N, 4, 4) or (4, 4) torch or np
    """
    if torch.is_tensor(input):
        if len(input.shape) == 3:
            output = torch.cat([input, torch.zeros_like(input[:, 0:1])], dim=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = torch.cat([input, torch.tensor([[0,0,0,1]], dtype=input.dtype, device=input.device)], dim=0)  # (4, 4)
    else:
        if len(input.shape) == 3:
            output = np.concatenate([input, np.zeros_like(input[:, 0:1])], axis=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = np.concatenate([input, np.array([[0,0,0,1]], dtype=input.dtype)], axis=0)  # (4, 4)
            output[3, 3] = 1.0
    return output


def vec2skew(v):
    """
    :param v:  (3, ) torch tensor
    :return:   (3, 3)
    """
    zero = torch.zeros(1, dtype=torch.float32, device=v.device)
    skew_v0 = torch.cat([ zero,    -v[2:3],   v[1:2]])  # (3, 1)
    skew_v1 = torch.cat([ v[2:3],   zero,    -v[0:1]])
    skew_v2 = torch.cat([-v[1:2],   v[0:1],   zero])
    skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=0)  # (3, 3)
    return skew_v  # (3, 3)


def Exp(r):
    """so(3) vector to SO(3) matrix
    :param r: (3, ) axis-angle, torch tensor
    :return:  (3, 3)
    """
    skew_r = vec2skew(r)  # (3, 3)
    norm_r = r.norm() + 1e-15
    eye = torch.eye(3, dtype=torch.float32, device=r.device)
    R = eye + (torch.sin(norm_r) / norm_r) * skew_r + ((1 - torch.cos(norm_r)) / norm_r**2) * (skew_r @ skew_r)
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
    k_matrix = torch.cat([
        torch.cat([f_sub_mat, torch.reshape(torch.stack([cx, cy]), shape=(2, 1))], dim=1),
        torch.reshape(torch.stack([zero, zero, one]), shape=(1, 3))
    ], dim=0)
    k_matrix = torch.cat([k_matrix, torch.reshape(torch.stack([zero, zero, zero]), shape=(3, 1))], dim=1)
    k_matrix = torch.cat([k_matrix, torch.reshape(torch.stack([zero, zero, zero, one]), shape=(1, 4))], dim=0)
    return k_matrix


class LearnPose(nn.Module):
    def __init__(self, num_cams, learn_R, learn_t, init_c2w=None):
        """
        :param num_cams:
        :param learn_R:  True/False
        :param learn_t:  True/False
        :param init_c2w: (N, 4, 4) torch tensor
        """
        super(LearnPose, self).__init__()
        self.num_cams = num_cams
        self.init_c2w = None
        if init_c2w is not None:
            self.init_c2w = nn.Parameter(init_c2w, requires_grad=False)

        self.r = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_R)  # (N, 3)
        if init_c2w is not None:
            self.t = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_t)  # (N, 3)
        else:
            self.t = nn.Parameter(
                torch.tensor([[0.0, 0.0, -2.0] for _ in range(num_cams)], dtype=torch.float32), requires_grad=learn_t
            )  # (N, 3)

    def forward(self, cam_id):
        r = self.r[cam_id]  # (3, ) axis-angle
        t = self.t[cam_id]  # (3, )
        c2w = make_c2w(r, t)  # (4, 4)

        # learn a delta pose between init pose and target pose, if a init pose is provided
        if self.init_c2w is not None:
            c2w = c2w @ self.init_c2w[cam_id]

        return c2w


class LearnFocal(nn.Module):
    def __init__(self, H, W, req_grad, fx_only, order=2, init_focal=None, init_center=None):
        super(LearnFocal, self).__init__()
        self.H = H
        self.W = W
        self.fx_only = fx_only  # If True, output [fx, fx]. If False, output [fx, fy]
        self.order = order  # check our supplementary section.

        if self.fx_only:
            if init_focal is None:
                self.fx = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=req_grad)  # (1, )
            else:
                if self.order == 2:
                    # a**2 * W = fx  --->  a**2 = fx / W
                    coe_x = torch.tensor(np.sqrt(init_focal / float(W)), requires_grad=False).float()
                elif self.order == 1:
                    # a * W = fx  --->  a = fx / W
                    coe_x = torch.tensor(init_focal / float(W), requires_grad=False).float()
                else:
                    print('Focal init order need to be 1 or 2. Exit')
                    exit()
                self.fx = nn.Parameter(coe_x, requires_grad=req_grad)  # (1, )
        else:
            if init_focal is None:
                self.fx = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=req_grad)  # (1, )
                self.fy = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=req_grad)  # (1, )
            else:
                if self.order == 2:
                    # a**2 * W = fx  --->  a**2 = fx / W
                    coe_x = torch.tensor(np.sqrt(init_focal / float(W)), requires_grad=False).float()
                    coe_y = torch.tensor(np.sqrt(init_focal / float(H)), requires_grad=False).float()
                elif self.order == 1:
                    # a * W = fx  --->  a = fx / W
                    coe_x = torch.tensor(init_focal / float(W), requires_grad=False).float()
                    coe_y = torch.tensor(init_focal / float(H), requires_grad=False).float()
                else:
                    print('Focal init order need to be 1 or 2. Exit')
                    exit()
                self.fx = nn.Parameter(coe_x, requires_grad=req_grad)  # (1, )
                self.fy = nn.Parameter(coe_y, requires_grad=req_grad)  # (1, )

        if init_center is None:
            self.cx = nn.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=req_grad)  # (1, )
            self.cy = nn.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=req_grad)  # (1, )
        else:
            self.cx = nn.Parameter(
                torch.tensor(init_center[0], requires_grad=False).float(), requires_grad=req_grad
            )  # (1, )
            self.cy = nn.Parameter(
                torch.tensor(init_center[1], requires_grad=False).float(), requires_grad=req_grad
            )  # (1, )

    def forward(self, inverse=False):
        if self.fx_only:
            if self.order == 2:
                fx = self.fx ** 2 * self.W
                fy = self.fx ** 2 * self.W
                k_matrix = make_calib(fx, fy, self.cx, self.cy)
            else:
                fx = self.fx * self.W
                fy = self.fx * self.W
                k_matrix = make_calib(fx, fy, self.cx, self.cy)
        else:
            if self.order == 2:
                fx = self.fx**2 * self.W
                fy = self.fy**2 * self.H
                k_matrix = make_calib(fx, fy, self.cx, self.cy)
            else:
                fx = self.fx * self.W
                fy = self.fy * self.H
                k_matrix = make_calib(fx, fy, self.cx, self.cy)
        if inverse:
            return torch.inverse(k_matrix)
        else:
            return k_matrix
