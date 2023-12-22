import stable_baselines3 as sb3

import torch
import os
import time
import datetime
import imageio
import argparse
import numpy as np
import joblib

from robot.robot_env import RobotEnv
from trackers.color_tracker import ColorTracker
from helpers.pointclouds import crop_points
from helpers.transformations import quat_to_euler

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    # experiment
    parser.add_argument("--exp", type=str)
    parser.add_argument("--save_dir", type=str, default="data")
    
    # hardware
    parser.add_argument("--dof", type=int, default=4, choices=[2, 3, 4, 6])
    parser.add_argument("--robot_type", type=str, default="panda", choices=["panda", "fr3"])
    parser.add_argument("--ip_address", type=str, default="172.16.0.1", choices=[None, "localhost", "172.16.0.1"])
    parser.add_argument("--camera_model", type=str, default="realsense", choices=["realsense", "zed"])
    
    # training
    parser.add_argument("--max_episode_length", type=int, default=20)
    parser.add_argument("--gpu_id", type=int, default=0)

    args = parser.parse_args()

    args.exp = "test"
    assert args.exp is not None, "Specify --exp"
    device = torch.device(("cuda:" + str(args.gpu_id)) if args.gpu_id >= 0. and torch.cuda.is_available() else "cpu")
    
    control_hz = 10
    env = RobotEnv(
        control_hz=control_hz,
        DoF=args.dof,
        robot_type=args.robot_type,
        gripper=True,
        ip_address=args.ip_address,
        camera_model=args.camera_model,
        max_lin_vel=2.0,
        max_rot_vel=5.0,
        max_path_length=args.max_episode_length,
    )

    tracker = ColorTracker(outlier_removal=True)
    tracker.reset()
    # define workspace
    crop_min = [0.0, -0.6, -0.1]
    crop_max = [0.7, 0.6, 0.5]
 
    obs = env.reset()
    
    imgs = []
    actions = np.ones(env.action_shape)

    # run for 25 steps to get filtered estimate
    for i in range(25):
        obs_dict = env.get_images_and_points()
        rgbs, points = [], []
        for key in obs_dict.keys():
            rgbs.append(obs_dict[key]['rgb'])
            points.append(obs_dict[key]['points'])
        tracked_points = tracker.track_multiview(rgbs, points, color="red", show=False)
        cropped_points = crop_points(tracked_points, crop_min=crop_min, crop_max=crop_max)
        
        # WARNING: only use filter when position is fixed!
        rod_pose = tracker.get_rod_pose(cropped_points, lowpass_filter=True, cutoff_freq=1, control_hz=control_hz, show=False)
        time.sleep(0.1)

    rod_angle = quat_to_euler(rod_pose[3:]).copy()
    print(env._robot.get_ee_angle(), rod_angle)

    # MOVE ABOVE ROD
    target_pose = rod_pose.copy()
    # set fixed height
    target_pose[2] = 0.4
    # set rod z angle + offset
    target_pose[3:6] = rod_angle + np.pi/4
    # overwrite x,y angle w/ gripper default
    target_pose[3:5] = env._default_angle[:2]
    # set dummy gripper
    target_pose[-1] = 0

    # update pose
    env._robot.update_pose(target_pose, blocking=True)

    # MOVE DOWN
    target_pose[2] = 0.11
    env._robot.update_pose(target_pose, blocking=True)
    
    # PICK
    env._robot.update_gripper(1., velocity=False, blocking=True)

    # MOVE UP
    target_pose[2] = 0.2
    env._robot.update_pose(target_pose, blocking=True)

    # MOVE DOWN
    target_pose[2] = 0.10
    env._robot.update_pose(target_pose, blocking=True)
    
    # DROP
    env._robot.update_gripper(0., velocity=False, blocking=True)

    # MOVE UP
    target_pose[2] = 0.2
    env._robot.update_pose(target_pose, blocking=True)
    
    imgs.append(env.render())

    env.reset()

    imageio.mimsave("test_rollout.gif", np.stack(imgs), duration=3)
