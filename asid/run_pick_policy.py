import argparse
import datetime
import os
import time

import imageio
import joblib
import numpy as np
import torch

from perception.trackers.color_tracker import ColorTracker
from robot.robot_env import RobotEnv
from utils.pointclouds import crop_points
from utils.transformations import euler_to_rmat, quat_to_euler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # experiment
    parser.add_argument("--exp", type=str)
    parser.add_argument("--save_dir", type=str, default="data")

    # hardware
    parser.add_argument("--dof", type=int, default=4, choices=[2, 3, 4, 6])
    parser.add_argument(
        "--robot_type", type=str, default="panda", choices=["panda", "fr3"]
    )
    parser.add_argument(
        "--ip_address",
        type=str,
        default="localhost",
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
    env = RobotEnv(
        control_hz=control_hz,
        DoF=args.dof,
        robot_type=args.robot_type,
        gripper=True,
        ip_address=args.ip_address,
        camera_model=args.camera_model,
        max_path_length=args.max_episode_length,
    )

    tracker = ColorTracker(outlier_removal=False)
    tracker.reset()
    # define workspace
    crop_min = [0.0, -0.6, -0.1]
    crop_max = [0.7, 0.6, 0.5]

    obs = env.reset()

    # # MOVE ABOVE TOOL
    # target_pose = torch.tensor(
    #     [0.49, 0.34, 0.2079993, -3.14152481, -0.04132315, -0.81102356]
    # )
    # target_pose[3:] = torch.tensor(env._default_angle)
    # target_pose[2] = 0.23
    # env._robot.update_pose(target_pose, blocking=True)

    # # MOVE DOWN
    # target_pose[2] = 0.20
    # env._robot.update_pose(target_pose, blocking=True)

    # # MOVE DOWN
    # target_pose[2] = 0.19
    # env._robot.update_pose(target_pose, blocking=True)

    # # GRASP
    # env._robot.update_gripper(1.0, velocity=False, blocking=True)

    # # MOVE UP
    # target_pose[2] = 0.3
    # env._robot.update_pose(target_pose, blocking=True)

    imgs = []
    actions = np.ones(env.action_shape)

    # run for 25 steps to eliminate realsense noise or get filtered estimate
    for i in range(25):
        obs_dict = env.get_images_and_points()
        rgbs, points = [], []
        for key in obs_dict.keys():
            rgbs.append(obs_dict[key]["rgb"])
            points.append(obs_dict[key]["points"])
        tracked_points = tracker.track_multiview(rgbs, points, color="red", show=False)
        cropped_points = crop_points(
            tracked_points, crop_min=crop_min, crop_max=crop_max
        )

        rod_pose = tracker.get_rod_pose(
            cropped_points,
            lowpass_filter=False,
            cutoff_freq=1,
            control_hz=control_hz,
            show=False,
        )
        time.sleep(0.1)

    rod_angle = quat_to_euler(rod_pose[3:]).copy()
    print(env._robot.get_ee_angle(), rod_angle)

    # MOVE ABOVE ROD
    target_pose = rod_pose.copy()
    # set fixed height
    target_pose[2] = 0.4
    # set rod z angle + z offset for franka EE
    target_pose[3:6] = rod_angle
    target_pose[5] += np.pi / 4
    # overwrite x,y angle w/ gripper default
    target_pose[3:5] = env._default_angle[:2]
    # set dummy gripper
    target_pose[-1] = 0

    # get rotation matrix from rod angle
    rod_angle_z = np.zeros(3)
    rod_angle_z[2] = rod_angle[2]
    rmat = euler_to_rmat(rod_angle_z)
    # transform rod pos to origin
    rod_pos_origin = target_pose[:3] @ np.linalg.inv(rmat)
    # add displacement
    # lenght of rod = 0.15 | +/- 0.07
    # + away from robot, - towards robot
    rod_pos_origin[0] += +0.08
    # transform back
    rod_pos_new = rod_pos_origin @ rmat
    # overwrite x,y
    target_pose[:2] = rod_pos_new[:2]

    env._robot.update_pose(target_pose, blocking=True)
    imgs.append(env.render())

    # MOVE DOWN ABOVE
    target_pose[2] = 0.17
    env._robot.update_pose(target_pose, blocking=True)
    imgs.append(env.render())

    # MOVE DOWN
    target_pose[2] = 0.12
    env._robot.update_pose(target_pose, blocking=True)
    imgs.append(env.render())

    # PICK
    env._robot.update_gripper(1.0, velocity=False, blocking=True)
    imgs.append(env.render())

    # MOVE UP
    target_pose[2] = 0.3
    env._robot.update_pose(target_pose, blocking=True)
    imgs.append(env.render())

    # MOVE TO YELLOW BLOCK
    target_pose = torch.tensor(
        [0.28793925, -0.35449123, 0.12003776, 3.07100117, 0.01574947, -0.80526758]
        # [0.28867856, -0.40134683, 0.11756707, 3.13773595, 0.0078624, -0.70369389]
    )
    target_pose[2] = 0.3
    env._robot.update_pose(target_pose, blocking=True)
    imgs.append(env.render())

    # MOVE DOWN + yellow block height + margin
    target_pose[2] = 0.12 + 0.05 + 0.02
    env._robot.update_pose(target_pose, blocking=True)
    imgs.append(env.render())

    # DROP
    env._robot.update_gripper(0.0, velocity=False, blocking=True)
    imgs.append(env.render())

    # MOVE UP
    target_pose[2] = 0.3
    env._robot.update_pose(target_pose, blocking=True)
    imgs.append(env.render())

    env.reset()

    imageio.mimsave("test_rollout.gif", np.stack(imgs), duration=3)
