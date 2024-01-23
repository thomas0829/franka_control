import argparse
import datetime
import os
import time

import imageio
import joblib
import numpy as np
import torch
from torchcontrol.policies import CartesianImpedanceControl

from helpers.pointclouds import crop_points
from helpers.transformations import euler_to_quat, quat_to_euler
from robot.robot_env import RobotEnv
from trackers.color_tracker import ColorTracker

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # experiment
    parser.add_argument("--exp", type=str)
    parser.add_argument("--save_dir", type=str, default="data")

    # hardware
    parser.add_argument("--dof", type=int, default=3, choices=[2, 3, 4, 6])
    parser.add_argument(
        "--robot_type", type=str, default="panda", choices=["panda", "fr3"]
    )
    parser.add_argument(
        "--ip_address",
        type=str,
        default=None,
        choices=[None, "localhost", "172.16.0.1"],
    )
    parser.add_argument(
        "--camera_model", type=str, default="realsense", choices=["realsense", "zed"]
    )

    # training
    parser.add_argument("--max_episode_length", type=int, default=20)
    parser.add_argument("--gpu_id", type=int, default=0)

    args = parser.parse_args()

    args.exp = "test"
    assert args.exp is not None, "Specify --exp"
    device = torch.device(
        ("cuda:" + str(args.gpu_id))
        if args.gpu_id >= 0.0 and torch.cuda.is_available()
        else "cpu"
    )

    control_hz = 10

    cfg = {
        "control_hz": control_hz,
        "DoF": args.dof,
        "robot_type": args.robot_type,
        "gripper": False,
        "ip_address": args.ip_address,
        "camera_model": args.camera_model,
        "camera_resolution": (480, 480),
        "max_path_length": args.max_episode_length,
        "model_name": "rod_franka",
        "on_screen_rendering": False,
    }

    env = RobotEnv(**cfg)
    from robot.sim.mujoco.asid_wrapper import ASIDWrapper
    # from robot.real.asid_wrapper import ASIDWrapper

    env = ASIDWrapper(
        env
    )
    env.reset()

    seed = 0
    env.seed(seed)
    env.create_exp_reward(cfg, seed)
    rew = env.compute_reward(np.zeros(args.dof))

    obs = env.reset()

    imgs = []
    for i in range(20):
        
        # ACT
        actions = np.array([0.0, 1.0, 0.0])

        next_obs, rew, done, _ = env.step(actions)
        print("reward", rew)
        imgs.append(env.render())

    env.reset()

    imageio.mimsave("test_rollout.gif", np.stack(imgs), duration=3)
    import matplotlib.pyplot as plt