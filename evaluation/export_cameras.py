import argparse
import os

import cv2
import numpy as np
import torch
from pyhocon import ConfigFactory

from models.camera import LearnFocal, LearnPose


def export_learned_cameras(conf):
    # get scene metadata
    base_exp_dir = conf["general.base_exp_dir"]
    data_dir = os.path.join(conf["dataset.data_dir"], "image")
    images_list = os.listdir(data_dir)
    image_sample = cv2.imread(os.path.join(data_dir, images_list[0]))
    H, W = image_sample.shape[0], image_sample.shape[1]
    init_pose = conf["dataset.init_pose"]

    # initialize camera networks
    if init_pose:
        pose_network = LearnPose(
            num_cams=len(images_list),
            init_c2w=torch.stack(
                [torch.zeros((4, 4)) for _ in range(len(images_list))], dim=0
            ),
        ).to("cuda")
    else:
        pose_network = LearnPose(num_cams=len(images_list), init_c2w=None).to("cuda")
    intrinsic_network = LearnFocal(H=H, W=W, req_grad=False, fx_only=False).to("cuda")

    # load camera networks weights
    model_list_raw = os.listdir(os.path.join(base_exp_dir, "checkpoints"))
    model_list = []
    for model_name in model_list_raw:
        if model_name[-3:] == "pth":
            model_list.append(model_name)
    model_list.sort()
    checkpoint = torch.load(
        os.path.join(base_exp_dir, "checkpoints", model_list[-1]), map_location="cuda"
    )

    intrinsic_network.load_state_dict(checkpoint["intrinsic_network"])
    pose_network.load_state_dict(checkpoint["pose_network"])

    # export learned camera parameters
    camera_parameters = {}
    for cam_id in range(len(images_list)):
        camera_parameters[f"pose_mat_{cam_id}"] = (
            pose_network(cam_id).detach().cpu().numpy()
        )
        camera_parameters[f"intrinsic_mat_{cam_id}"] = (
            intrinsic_network().detach().cpu().numpy()
        )
    np.savez(os.path.join(base_exp_dir, "camera.npz"), **camera_parameters)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, default="./configs/ba_no_poses.conf")
    parser.add_argument("--case", type=str, default="")

    args = parser.parse_args()

    f = open(args.conf)
    conf_text = f.read()
    conf_text = conf_text.replace("CASE_NAME", args.case)
    f.close()

    conf = ConfigFactory.parse_string(conf_text)

    export_learned_cameras(conf)
