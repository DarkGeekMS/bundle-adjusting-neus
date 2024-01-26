import logging
import os
from shutil import copyfile

import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
import wandb
from pyhocon import ConfigFactory
from tqdm import tqdm

from dataset.dataset import Dataset
from evaluation.camera_pose_visualizer import CameraPoseVisualizer
from losses.point_cloud_loss import transform_pixel_to_world, visualize_point_cloud
from losses.ssi_depth_loss import ScaleAndShiftInvariantLoss
from models.fields import NeRF, RenderingNetwork, SDFNetwork, SingleVarianceNetwork
from models.renderer import NeuSRenderer
from utils.camera_utils import scale_camera


class Runner:
    def __init__(self, conf_path, mode="train", case="CASE_NAME", is_continue=False):
        self.device = torch.device("cuda")

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace("CASE_NAME", case)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf["dataset.data_dir"] = self.conf["dataset.data_dir"].replace(
            "CASE_NAME", case
        )
        self.base_exp_dir = self.conf["general.base_exp_dir"]
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.exp_name = self.conf["general.exp_name"]
        self.dataset = Dataset(self.conf["dataset"])
        self.iter_step = 0

        # Training parameters
        self.end_iter = self.conf.get_int("train.end_iter")
        self.save_freq = self.conf.get_int("train.save_freq")
        self.report_freq = self.conf.get_int("train.report_freq")
        self.val_freq = self.conf.get_int("train.val_freq")
        self.val_mesh_freq = self.conf.get_int("train.val_mesh_freq")
        self.batch_size = self.conf.get_int("train.batch_size")
        self.validate_resolution_level = self.conf.get_int(
            "train.validate_resolution_level"
        )
        self.learning_rate = self.conf.get_float("train.learning_rate")
        self.learning_rate_alpha = self.conf.get_float("train.learning_rate_alpha")
        self.use_white_bkgd = self.conf.get_bool("train.use_white_bkgd")
        self.warm_up_end = self.conf.get_float("train.warm_up_end", default=0.0)
        self.anneal_end = self.conf.get_float("train.anneal_end", default=0.0)
        self.use_masked_loss = self.conf.get_bool("train.use_masked_loss")
        self.use_ssi_depth_loss = self.conf.get_bool("train.use_ssi_depth_loss")
        self.igr_weight = self.conf.get_float("train.igr_weight")
        self.mask_weight = self.conf.get_float("train.mask_weight")
        self.depth_weight = self.conf.get_float("train.depth_weight")
        self.phase_delims = self.conf.get_list("train.phase_delim")
        self.pc_weights = self.conf.get_list("train.pc_weight")
        self.feat_weights = self.conf.get_list("train.feat_weight")
        self.depth_from_inside_only_s = self.conf.get_list(
            "train.depth_from_inside_only"
        )
        self.object_mask_type = self.conf.get_string("train.object_mask_type")
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []

        # SSI Depth Loss
        self.depth_loss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1)

        # Initialize WandB run
        wandb.init(project="sparse_bundle_neus", config=vars(self.conf))
        wandb.run.name = self.exp_name
        wandb.run.save()

        # Networks
        params_to_train = []
        self.nerf_outside = NeRF(**self.conf["model.nerf"]).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf["model.sdf_network"]).to(self.device)
        self.deviation_network = SingleVarianceNetwork(
            **self.conf["model.variance_network"]
        ).to(self.device)
        self.color_network = RenderingNetwork(
            **self.conf["model.rendering_network"]
        ).to(self.device)
        params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())

        self.intrinsic_optimizer = torch.optim.Adam(
            self.dataset.intrinsic_network.parameters(), lr=self.learning_rate
        )
        self.pose_optimizer = torch.optim.Adam(
            self.dataset.pose_network.parameters(), lr=self.learning_rate
        )
        self.distortion_optimizer = torch.optim.Adam(
            self.dataset.distortion_network.parameters(), lr=self.learning_rate
        )
        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        self.renderer = NeuSRenderer(
            self.nerf_outside,
            self.sdf_network,
            self.deviation_network,
            self.color_network,
            **self.conf["model.neus_renderer"]
        )

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, "checkpoints"))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == "pth" and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info("Find checkpoint: {}".format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        # Backup codes and configs for debug
        if self.mode[:5] == "train":
            self.file_backup()

    def train(self):
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        image_perm = self.get_image_perm()

        for iter_i in tqdm(range(res_step)):
            data, feat_input, pc_input = self.dataset.gen_random_rays_at(
                image_perm[self.iter_step % len(image_perm)], self.batch_size
            )

            rays_o, rays_d, true_rgb, mask, depth = (
                data[:, :3],
                data[:, 3:6],
                data[:, 6:9],
                data[:, 9:10],
                data[:, 10:11],
            )
            near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

            for attr in [
                "depth_cams",
                "size",
                "center",
                "feat",
                "feat_src",
                "cam",
                "src_cams",
                "rays_v_norm",
            ]:
                feat_input[attr] = feat_input[attr].cuda()
            for attr in ["H", "W"]:
                feat_input[attr] = feat_input[attr]

            for attr in ["ref_cam", "ref_depth", "src_cams", "src_depths", "pc_pixels"]:
                pc_input[attr] = pc_input[attr].cuda()

            background_rgb = None
            if self.use_white_bkgd:
                background_rgb = torch.ones([1, 3])

            if self.mask_weight > 0.0:
                mask = (mask > 0.5).float()
            else:
                mask = torch.ones_like(mask)

            mask_sum = mask.sum() + 1e-5
            train_phase = self.iter_step / self.end_iter
            render_out = self.renderer.render(
                rays_o,
                rays_d,
                near,
                far,
                background_rgb=background_rgb,
                cos_anneal_ratio=self.get_cos_anneal_ratio(),
                pc_input=pc_input,
                feat_input=feat_input,
                depth_from_inside_only=self.get_param_in_phase(
                    self.depth_from_inside_only_s, train_phase
                ),
                object_mask_type=self.object_mask_type,
            )

            color_fine = render_out["color_fine"]
            s_val = render_out["s_val"]
            cdf_fine = render_out["cdf_fine"]
            gradient_error = render_out["gradient_error"]
            weight_max = render_out["weight_max"]
            weight_sum = render_out["weight_sum"]
            depth_pred = render_out["depth_fine"]
            feat_loss = render_out["feat_loss"]
            pc_loss = render_out["pc_loss"]

            # Loss
            color_error = (color_fine - true_rgb) * mask
            color_fine_loss = (
                F.l1_loss(color_error, torch.zeros_like(color_error), reduction="sum")
                / mask_sum
            )
            psnr = 20.0 * torch.log10(
                1.0
                / (
                    ((color_fine - true_rgb) ** 2 * mask).sum() / (mask_sum * 3.0)
                ).sqrt()
            )

            eikonal_loss = gradient_error

            mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)

            if self.use_masked_loss:
                depth_loss = self.get_depth_loss(depth_pred, depth, mask)
            else:
                depth_loss = self.get_depth_loss(
                    depth_pred, depth, torch.ones_like(mask)
                )

            pc_weight = self.get_param_in_phase(self.pc_weights, train_phase)
            feat_weight = self.get_param_in_phase(self.feat_weights, train_phase)

            loss = (
                color_fine_loss
                + eikonal_loss * self.igr_weight
                + mask_loss * self.mask_weight
                + feat_loss * feat_weight
                + depth_loss * self.depth_weight
                + pc_loss * pc_weight
            )

            self.optimizer.zero_grad()
            self.intrinsic_optimizer.zero_grad()
            self.pose_optimizer.zero_grad()
            self.distortion_optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.intrinsic_optimizer.step()
            self.pose_optimizer.step()
            self.distortion_optimizer.step()

            self.iter_step += 1

            self.sdf_network.progress.data.fill_(self.iter_step / self.end_iter)
            self.color_network.progress.data.fill_(self.iter_step / self.end_iter)
            self.nerf_outside.progress.data.fill_(self.iter_step / self.end_iter)

            wandb.log(
                {
                    "Loss/loss": loss,
                    "Loss/color_loss": color_fine_loss,
                    "Loss/eikonal_loss": eikonal_loss,
                    "Loss/feat_loss": feat_loss,
                    "Loss/depth_loss": depth_loss,
                    "Loss/pc_loss": pc_loss,
                    "Statistics/s_val": s_val.mean(),
                    "Statistics/cdf": (cdf_fine[:, :1] * mask).sum() / mask_sum,
                    "Statistics/weight_max": (weight_max * mask).sum() / mask_sum,
                    "Statistics/psnr": psnr,
                }
            )

            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                print(
                    "iter:{:8>d} loss={} color_loss={} eikonal_loss={} feat_loss={} depth_loss={} pc_loss={} psnr={} lr={}".format(
                        self.iter_step,
                        loss,
                        color_fine_loss,
                        eikonal_loss,
                        feat_loss,
                        depth_loss,
                        pc_loss,
                        psnr,
                        self.optimizer.param_groups[0]["lr"],
                    )
                )

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % self.val_freq == 0:
                idx = np.random.randint(self.dataset.n_images)
                self.validate_image(idx)
                self.validate_camera_poses()
                self.validate_point_cloud(idx)

            if self.iter_step % self.val_mesh_freq == 0:
                self.validate_mesh()

            self.update_learning_rate()

            if self.iter_step % len(image_perm) == 0:
                image_perm = self.get_image_perm()

    def get_depth_loss(self, depth_pred, depth_gt, mask):
        if self.use_ssi_depth_loss:
            depth_loss = self.depth_loss(
                depth_pred.reshape(1, 16, 32),
                depth_gt.reshape(1, 16, 32),
                mask.reshape(1, 16, 32),
            )
        else:
            depth_error = (depth_pred - depth_gt) * mask
            depth_loss = F.l1_loss(
                depth_error, torch.zeros_like(depth_error), reduction="sum"
            ) / (mask.sum() + 1e-5)
        return depth_loss

    def get_normal_loss(self, normal_pred, normal_gt, mask):
        normal_gt = torch.nn.functional.normalize(normal_gt, p=2, dim=-1)
        normal_pred = torch.nn.functional.normalize(normal_pred, p=2, dim=-1)
        l1 = torch.abs((normal_pred - normal_gt) * mask).sum(dim=-1).sum() / (
            mask.sum() + 1e-5
        )
        cos = (1.0 - torch.sum(normal_pred * normal_gt * mask, dim=-1)).sum() / (
            mask.sum() + 1e-5
        )
        return l1 + cos

    def get_param_in_phase(self, param_list, phase):
        if phase < self.phase_delims[0]:
            return param_list[0]
        elif phase < self.phase_delims[1]:
            return param_list[1]
        else:
            return param_list[2]

    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (
                self.end_iter - self.warm_up_end
            )
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (
                1 - alpha
            ) + alpha

        for g in self.optimizer.param_groups:
            g["lr"] = self.learning_rate * learning_factor

        for g in self.intrinsic_optimizer.param_groups:
            g["lr"] = self.learning_rate * learning_factor

        for g in self.pose_optimizer.param_groups:
            g["lr"] = self.learning_rate * learning_factor

        for g in self.distortion_optimizer.param_groups:
            g["lr"] = self.learning_rate * learning_factor

    def file_backup(self):
        dir_lis = self.conf["general.recording"]
        os.makedirs(os.path.join(self.base_exp_dir, "recording"), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, "recording", dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == ".py":
                    copyfile(
                        os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name)
                    )

        copyfile(
            self.conf_path, os.path.join(self.base_exp_dir, "recording", "config.conf")
        )

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(
            os.path.join(self.base_exp_dir, "checkpoints", checkpoint_name),
            map_location=self.device,
        )
        self.dataset.intrinsic_network.load_state_dict(checkpoint["intrinsic_network"])
        self.dataset.pose_network.load_state_dict(checkpoint["pose_network"])
        self.dataset.distortion_network.load_state_dict(
            checkpoint["distortion_network"]
        )
        self.nerf_outside.load_state_dict(checkpoint["nerf"])
        self.sdf_network.load_state_dict(checkpoint["sdf_network_fine"])
        self.deviation_network.load_state_dict(checkpoint["variance_network_fine"])
        self.color_network.load_state_dict(checkpoint["color_network_fine"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.intrinsic_optimizer.load_state_dict(checkpoint["intrinsic_optimizer"])
        self.pose_optimizer.load_state_dict(checkpoint["pose_optimizer"])
        self.distortion_optimizer.load_state_dict(checkpoint["distortion_optimizer"])
        self.iter_step = checkpoint["iter_step"]

        logging.info("End")

    def save_checkpoint(self):
        checkpoint = {
            "nerf": self.nerf_outside.state_dict(),
            "sdf_network_fine": self.sdf_network.state_dict(),
            "variance_network_fine": self.deviation_network.state_dict(),
            "color_network_fine": self.color_network.state_dict(),
            "intrinsic_network": self.dataset.intrinsic_network.state_dict(),
            "pose_network": self.dataset.pose_network.state_dict(),
            "distortion_network": self.dataset.distortion_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "intrinsic_optimizer": self.intrinsic_optimizer.state_dict(),
            "pose_optimizer": self.pose_optimizer.state_dict(),
            "distortion_optimizer": self.distortion_optimizer.state_dict(),
            "iter_step": self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, "checkpoints"), exist_ok=True)
        torch.save(
            checkpoint,
            os.path.join(
                self.base_exp_dir,
                "checkpoints",
                "ckpt_{:0>6d}.pth".format(self.iter_step),
            ),
        )

    def validate_image(self, idx, resolution_level=-1):
        print("Validate: iter: {}, camera: {}".format(self.iter_step, idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d = self.dataset.gen_rays_at(
            idx, resolution_level=resolution_level
        )
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        out_normal_fine = []
        out_depth_fine = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(
                rays_o_batch,
                rays_d_batch,
                near,
                far,
                cos_anneal_ratio=self.get_cos_anneal_ratio(),
                background_rgb=background_rgb,
            )

            def feasible(key):
                return (key in render_out) and (render_out[key] is not None)

            if feasible("color_fine"):
                out_rgb_fine.append(render_out["color_fine"].detach().cpu().numpy())
            if feasible("normal_fine"):
                out_normal_fine.append(render_out["normal_fine"].detach().cpu().numpy())
            if feasible("depth_fine"):
                out_depth_fine.append(render_out["depth_fine"].detach().cpu().numpy())
            del render_out

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (
                np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256
            ).clip(0, 255)

        depth_fine = None
        if len(out_depth_fine) > 0:
            depth_fine = np.concatenate(out_depth_fine, axis=0).reshape([H, W, 1, -1])
            depth_fine[depth_fine < 0] = 0

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = np.linalg.inv(
                self.dataset.pose_network(idx)[:3, :3].detach().cpu().numpy()
            )
            normal_img = (
                np.matmul(rot[None, :, :], normal_img[:, :, None]).reshape(
                    [H, W, 3, -1]
                )
                * 128
                + 128
            ).clip(0, 255)

        os.makedirs(os.path.join(self.base_exp_dir, "validations_fine"), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, "normals"), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, "depths"), exist_ok=True)

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(
                    os.path.join(
                        self.base_exp_dir,
                        "validations_fine",
                        "{:0>8d}_{}_{}.png".format(self.iter_step, i, idx),
                    ),
                    np.concatenate(
                        [
                            img_fine[..., i],
                            self.dataset.image_at(
                                idx, resolution_level=resolution_level
                            ),
                        ]
                    ),
                )
            if len(out_normal_fine) > 0:
                cv.imwrite(
                    os.path.join(
                        self.base_exp_dir,
                        "normals",
                        "{:0>8d}_{}_{}.png".format(self.iter_step, i, idx),
                    ),
                    normal_img[..., i],
                )
            if len(out_depth_fine) > 0:
                cv.imwrite(
                    os.path.join(
                        self.base_exp_dir,
                        "depths",
                        "{:0>8d}_{}_{}.png".format(self.iter_step, i, idx),
                    ),
                    (255 * depth_fine[..., i] / depth_fine[..., i].max()).astype(
                        np.uint8
                    ),
                )

    def validate_camera_poses(self):
        os.makedirs(os.path.join(self.base_exp_dir, "cameras"), exist_ok=True)
        visualizer = CameraPoseVisualizer([-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0])
        for cam_idx in range(self.dataset.n_images):
            gt_pose = self.dataset.pose_all_gt[cam_idx].cpu().numpy()
            pred_pose = self.dataset.pose_network(cam_idx).detach().cpu().numpy()
            visualizer.extrinsic2pyramid(gt_pose, "g", 0.5)
            visualizer.extrinsic2pyramid(pred_pose, "r", 0.5)
        visualizer.customize_legend()
        visualizer.save(
            os.path.join(
                self.base_exp_dir, "cameras", "{:0>8d}.png".format(self.iter_step)
            )
        )

    def validate_point_cloud(self, idx):
        os.makedirs(os.path.join(self.base_exp_dir, "point_clouds"), exist_ok=True)

        pc_pixels = self.dataset.pc_pixels

        depth = self.dataset.undistort_depth(
            idx, self.dataset.pc_scaled_depths[idx]
        ).view(1, -1, 1)
        cam = torch.stack(
            [self.dataset.pose_network(idx), self.dataset.intrinsic_network()], dim=0
        )
        cam = scale_camera(cam, self.dataset.pc_scale)
        camera_mat = torch.unsqueeze(cam[1], 0)
        world_mat = torch.unsqueeze(cam[0], 0)

        pc = transform_pixel_to_world(pc_pixels, depth, camera_mat, world_mat)

        pc = pc[0].detach().cpu().numpy()

        visualize_point_cloud(
            pc,
            os.path.join(
                self.base_exp_dir,
                "point_clouds",
                "{:0>8d}_{}.html".format(self.iter_step, idx),
            ),
        )

    def render_novel_image(self, idx_0, idx_1, ratio, resolution_level):
        """
        Interpolate view between two cameras.
        """
        rays_o, rays_d = self.dataset.gen_rays_between(
            idx_0, idx_1, ratio, resolution_level=resolution_level
        )
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(
                rays_o_batch,
                rays_d_batch,
                near,
                far,
                cos_anneal_ratio=self.get_cos_anneal_ratio(),
                background_rgb=background_rgb,
            )

            out_rgb_fine.append(render_out["color_fine"].detach().cpu().numpy())

            del render_out

        img_fine = (
            (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256)
            .clip(0, 255)
            .astype(np.uint8)
        )
        return img_fine

    def validate_mesh(self, world_space=False, resolution=64, threshold=0.0):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        vertices, triangles = self.renderer.extract_geometry(
            bound_min, bound_max, resolution=resolution, threshold=threshold
        )
        os.makedirs(os.path.join(self.base_exp_dir, "meshes"), exist_ok=True)

        if world_space:
            vertices = (
                vertices * self.dataset.scale_mats_np[0][0, 0]
                + self.dataset.scale_mats_np[0][:3, 3][None]
            )

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(
            os.path.join(
                self.base_exp_dir, "meshes", "{:0>8d}.ply".format(self.iter_step)
            )
        )

        logging.info("End")

    def interpolate_view(self, img_idx_0, img_idx_1):
        images = []
        n_frames = 60
        for i in range(n_frames):
            print(i)
            images.append(
                self.render_novel_image(
                    img_idx_0,
                    img_idx_1,
                    np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5,
                    resolution_level=4,
                )
            )
        for i in range(n_frames):
            images.append(images[n_frames - i - 1])

        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        video_dir = os.path.join(self.base_exp_dir, "render")
        os.makedirs(video_dir, exist_ok=True)
        h, w, _ = images[0].shape
        writer = cv.VideoWriter(
            os.path.join(
                video_dir,
                "{:0>8d}_{}_{}.mp4".format(self.iter_step, img_idx_0, img_idx_1),
            ),
            fourcc,
            30,
            (w, h),
        )

        for image in images:
            writer.write(image)

        writer.release()
