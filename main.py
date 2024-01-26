import argparse
import logging

import torch

from optimizers.ba_neus_optimizer import Runner


def run_optimization():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, default="./configs/ba_no_poses.conf")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--mcube_threshold", type=float, default=0.0)
    parser.add_argument("--is_continue", default=False, action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--case", type=str, default="")

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.mode, args.case, args.is_continue)

    if args.mode == "train":
        runner.train()
    elif args.mode == "validate_mesh":
        runner.validate_mesh(
            world_space=False, resolution=1024, threshold=args.mcube_threshold
        )
    elif args.mode.startswith(
        "interpolate"
    ):  # Interpolate views given two image indices
        _, img_idx_0, img_idx_1 = args.mode.split("_")
        img_idx_0 = int(img_idx_0)
        img_idx_1 = int(img_idx_1)
        runner.interpolate_view(img_idx_0, img_idx_1)


if __name__ == "__main__":
    print("Bundle-Adjusting NeuS")
    run_optimization()
