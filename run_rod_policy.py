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
    parser.add_argument("--dof", type=int, default=2, choices=[2, 3, 4, 6])
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
        gripper=False,
        ip_address=args.ip_address,
        camera_model=args.camera_model,
        max_lin_vel=1.0,
        max_rot_vel=1.0,
        max_path_length=args.max_episode_length,
    )

    tracker = ColorTracker(outlier_removal=True)
    tracker.reset()
    # define workspace
    crop_min = [0.0, -0.4, -0.1]
    crop_max = [0.5, 0.4, 0.5]
    
    # custom reset pose
    env._reset_joint_qpos = np.array(
            [
                0.02013862,
                0.50847548,
                -0.09224909,
                -2.36841345,
                0.1598147,
                2.88097692,
                0.63428867
            ]
        )
    obs = env.reset()

    # fixed z value (overwrites reset)
    env.ee_space.low[2] = 0.13
    env.ee_space.high[2] = 0.14
    
    # env._robot.update_command(_reset_joint_qpos, action_space="joint_position", blocking=True)
    # obs = env.get_observation()

    from gym.spaces import Box
    env.observation_space = Box(-np.ones(16), np.ones(16))

    model = sb3.PPO("MlpPolicy", env, device=device, n_steps=10, batch_size=48)
    model = model.load("policy.zip")

    imgs = []
    for i in range(args.max_episode_length):
        
        # TRACKING
        obs_dict = env.get_images_and_points()
        rgbs, points = [], []
        for key in obs_dict.keys():
            rgbs.append(obs_dict[key]['rgb'])
            points.append(obs_dict[key]['points'])
        tracked_points = tracker.track_multiview(rgbs, points, color="red", show=False)
        cropped_points = crop_points(tracked_points, crop_min=crop_min, crop_max=crop_max)
        rod_pose = tracker.get_rod_pose(cropped_points, lowpass_filter=True, cutoff_freq=1, control_hz=control_hz, show=False)

        # PREDICT
        obs_tmp = np.concatenate((obs["lowdim_ee"][:2], rod_pose, obs["lowdim_qpos"][:-1]))
        actions, _state = model.predict(obs_tmp, deterministic=False)

        # ACT
        # actions = np.random.uniform(-1, 1, size=env.action_shape)
        next_obs, rew, done, _ = env.step(actions)
        imgs.append(env.render())
        obs = next_obs

    env.reset()

    imageio.mimsave("test_rollout.gif", np.stack(imgs), duration=3)
