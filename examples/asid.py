import os
import time
import datetime
import argparse

import torch
import imageio
import numpy as np

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

    from robot.sim.vec_env.asid_vec import make_env
    env = make_env(
        cfg,
        seed=0,
        device_id=args.gpu_id,
        exp_reward=True,
        verbose=False,
    )

    obs = env.reset()

    imgs = []
    for i in range(20):
        actions = np.ones((args.dof)) * 0.1

        next_obs, rew, done, _ = env.step(actions)

        imgs.append(env.render())

    imageio.mimsave("test_rollout.gif", np.stack(imgs), duration=3)

    # num_workers = 1
    # from robot.sim.vec_env.asid_vec import make_env, make_vec_env
    # env = make_vec_env(
    #     cfg,
    #     num_workers=num_workers,
    #     seed=0,
    #     device_id=args.gpu_id,
    #     exp_reward=True,
    #     gymnasium=False,
    # )

    # obs = env.reset()

    # imgs = []
    # for i in range(20):
    #     actions = np.ones((num_workers, args.dof)) * 0.1

    #     next_obs, rew, done, _ = env.step(actions)

    #     imgs.append(env.render())

    # imageio.mimsave("test_rollout.gif", np.stack(imgs)[:,0], duration=3)
