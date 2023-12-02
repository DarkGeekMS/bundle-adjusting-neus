import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from glob import glob
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
from models.camera import LearnPose, LearnFocal, make_c2w
from models.distortion import LearnDistortion
from utils.features import load_pair, scale_camera, FeatExt
from utils.point_cloud import arange_pixels


# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
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


class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf

        self.data_dir = conf.get_string('data_dir')
        self.render_cameras_name = conf.get_string('render_cameras_name')
        self.object_cameras_name = conf.get_string('object_cameras_name')

        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)

        self.init_pose = conf.get_bool('init_pose', default=True)
        self.init_intrinsic = conf.get_bool('init_intrinsic', default=True)
        self.learn_pose = conf.get_bool('learn_pose', default=True)
        self.learn_intrinsic = conf.get_bool('learn_intrinsic', default=True)

        self.noise_magnitude = conf.get_float('noise_magnitude', default=0.1)
        self.trans_offset = conf.get_float('trans_offset', default=2.0)

        self.learn_scale = self.conf.get_bool('learn_scale')
        self.learn_shift = self.conf.get_bool('learn_shift')
        self.fix_scaleN = self.conf.get_bool('fix_scaleN')

        self.num_src = conf.get_int('num_sources', default=2)
        self.type_src = conf.get_string('type_sources', default='mvs')

        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
        self.camera_dict = camera_dict
        self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image/*.png')))
        self.n_images = len(self.images_lis)
        self.images_np = np.stack([cv.imread(im_name) for im_name in self.images_lis]) / 256.0
        self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.png')))
        self.masks_np = np.stack([cv.imread(im_name) for im_name in self.masks_lis]) / 256.0
        self.depth_lis = sorted(glob(os.path.join(self.data_dir, 'depth/*.npy')))
        self.depths_np = np.stack([np.load(im_name)[..., None] for im_name in self.depth_lis])

        # world_mat is a projection matrix from world to image
        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.scale_mats_np = []

        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_all = []
        self.pose_all = []
        self.pose_all_gt = []

        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            rot_noise = torch.randn(3, device='cpu') * self.noise_magnitude
            trans_noise = torch.randn(3, device='cpu') * self.noise_magnitude
            c2w_noise = make_c2w(rot_noise, trans_noise)
            self.pose_all.append(c2w_noise @ torch.from_numpy(pose).float())
            self.pose_all_gt.append(torch.from_numpy(pose).float())

        self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()  # [n_images, H, W, 3]
        self.masks = torch.from_numpy(self.masks_np.astype(np.float32)).cpu()  # [n_images, H, W, 3]
        self.depths = torch.from_numpy(self.depths_np.astype(np.float32)).cpu()  # [n_images, H, W, 1]
        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)   # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0].cpu()
        self.camera_center = [self.intrinsics_all[0][0, 2].cpu(), self.intrinsics_all[0][1, 2].cpu()]
        self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]
        self.pose_all_gt = torch.stack(self.pose_all_gt).to(self.device)  # [n_images, 4, 4]
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.image_pixels = self.H * self.W

        # Camera Networks
        if self.init_pose:
            self.pose_network = LearnPose(num_cams=len(self.pose_all), init_c2w=self.pose_all, trans_offset=self.trans_offset).to(self.device)
        else:
            self.pose_network = LearnPose(num_cams=len(self.pose_all), init_c2w=None, trans_offset=self.trans_offset).to(self.device)
        if self.init_intrinsic:
            self.intrinsic_network = LearnFocal(
                H=self.H, W=self.W, req_grad=self.learn_intrinsic, fx_only=False, init_focal=self.focal, init_center=self.camera_center
            ).to(self.device)
        else:
            self.intrinsic_network = LearnFocal(
                H=self.H, W=self.W, req_grad=self.learn_intrinsic, fx_only=False, init_focal=None, init_center=None
            ).to(self.device)

        # Depth Distortion Network
        self.distortion_network = LearnDistortion(
            self.n_images, learn_scale=self.learn_scale, learn_shift=self.learn_shift, fix_scaleN=self.fix_scaleN
        ).to(device=self.device)

        # Multi-view Features
        if self.type_src == 'mvs':
            self.pair = load_pair(f'{self.data_dir}/cam4feat/pair.txt')
        self.feat_img_scale = 2
        self.img_res = self.images.shape[-3:-1]

        self.rgb_2xd = torch.stack([
            F.interpolate(
                self.images[idx].reshape(-1, 3).permute(1, 0).view(1, 3, *self.img_res),
                size=(self.H * self.feat_img_scale, self.W * self.feat_img_scale),
                mode='bilinear', align_corners=False)[0]
            for idx in range(self.n_images)
        ], dim=0)  # [n_images, 3, 768, 768]

        mean = torch.tensor([0.485, 0.456, 0.406]).float().cpu()
        std = torch.tensor([0.229, 0.224, 0.225]).float().cpu()
        self.rgb_2xd = (self.rgb_2xd / 2 + 0.5 - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
        self.size = torch.from_numpy(self.scale_mats_np[0]).float()[0, 0] * 2
        self.center = torch.from_numpy(self.scale_mats_np[0]).float()[:3, 3]

        feat_ext = FeatExt().cuda()
        feat_ext.eval()
        for p in feat_ext.parameters():
            p.requires_grad = False
        feats = []
        for start_i in range(0, self.n_images):
            eval_batch = self.rgb_2xd[start_i:start_i + 1]
            feat2 = feat_ext(eval_batch.cuda())[2].detach().cpu()
            feats.append(feat2)
        self.feats = torch.cat(feats, dim=0)

        # Point Cloud
        self.pc_scale = 0.08
        self.pc_resolution = (int(self.H * self.pc_scale), int(self.W * self.pc_scale))
        self.pc_scaled_depths = [
            F.interpolate(torch.unsqueeze(torch.unsqueeze(depth[:, :, 0], 0), 0), self.pc_resolution, mode='nearest') for depth in self.depths
        ]
        self.pc_scaled_depths = torch.stack(self.pc_scaled_depths, dim=0)
        self.pc_pixels = arange_pixels(self.pc_resolution)

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0']
        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]

        print('Load data: End')

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        p = torch.matmul(self.intrinsic_network(inverse=True)[:3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(self.pose_network(img_idx)[:3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.pose_network(img_idx)[:3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
        depth = self.depths[img_idx][(pixels_y, pixels_x)]    # batch_size, 1
        depth = self.undistort_depth(img_idx, depth)          # batch_size, 1
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(self.intrinsic_network(inverse=True)[:3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v_norm = torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)   # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(self.pose_network(img_idx)[:3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_network(img_idx)[:3, 3].expand(rays_v.shape) # batch_size, 3
        if self.type_src == 'mvs':
            id = self.pair['id_list'][img_idx]
            src_ids = self.pair[id]['pair']
            src_idxs = [self.pair[src_id]['index'] for src_id in src_ids][:self.num_src]
        else:
            src_idxs = [img_idx.item() - 1, (img_idx.item() + 1) % self.n_images]
        depth_cam = torch.stack([self.pose_network(img_idx), self.intrinsic_network()], dim=0)
        cam = scale_camera(depth_cam, self.feat_img_scale)
        src_cams = torch.stack(
            [scale_camera(
                torch.stack([self.pose_network(i), self.intrinsic_network()], dim=0),
                self.feat_img_scale) for i in src_idxs]
        )
        feat_input = {
            'depth_cams': depth_cam,
            'size': self.size,
            'center': self.center,
            'feat': self.feats[img_idx],
            'feat_src': self.feats[src_idxs],
            'cam': cam,
            'src_cams': src_cams,
            'rays_v_norm': rays_v_norm,
            'H': self.H,
            'W': self.W
        }
        ref_cam = scale_camera(depth_cam, self.pc_scale)
        ref_depth = self.undistort_depth(img_idx, self.pc_scaled_depths[img_idx])
        src_cams = torch.stack(
            [scale_camera(
                torch.stack([self.pose_network(i), self.intrinsic_network()], dim=0),
                self.pc_scale) for i in src_idxs]
        )
        src_depths = torch.stack(
            [self.undistort_depth(i, self.pc_scaled_depths[i]) for i in src_idxs]
        )
        pc_input = {
            'ref_cam': ref_cam,
            'ref_depth': ref_depth,
            'src_cams': src_cams,
            'src_depths': src_depths,
            'pc_pixels': self.pc_pixels
        }
        return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1], depth.cpu()], dim=-1).cuda(), feat_input, pc_input  # batch_size, 11

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsic_network(inverse=True)[:3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        trans = self.pose_network(idx_0)[:3, 3] * (1.0 - ratio) + self.pose_network(idx_1)[:3, 3] * ratio
        pose_0 = self.pose_network(idx_0).detach().cpu().numpy()
        pose_1 = self.pose_network(idx_1).detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        near.relu_()
        far = mid + 1.0
        return near, far

    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)

    def undistort_depth(self, img_idx, depth):
        depth_copy = depth.clone().to(self.device)
        dist_scale, dist_shift = self.distortion_network(img_idx)
        depth_copy = depth_copy * dist_scale + dist_shift
        return depth_copy
