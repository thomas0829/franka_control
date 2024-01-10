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
    parser.add_argument("--dof", type=int, default=3, choices=[2, 3, 4, 6])
    parser.add_argument("--robot_type", type=str, default="fr3", choices=["panda", "fr3"])
    parser.add_argument("--ip_address", type=str, default=None, choices=[None, "localhost", "172.16.0.1"])
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
        camera_resolution=(480, 480),
        max_path_length=args.max_episode_length,
    )

    tracker = ColorTracker(outlier_removal=True)
    tracker.reset()
    # define workspace
    crop_min = [0.0, -0.4, -0.1]
    crop_max = [0.5, 0.4, 0.5]
    
    # custom reset pose
    obs = env.reset()
    
    # env._robot.update_command(_reset_joint_qpos, action_space="joint_position", blocking=True)
    # obs = env.get_observation()

    from gym.spaces import Box
    env.observation_space = Box(-np.ones(16), np.ones(16))

    model = sb3.PPO("MlpPolicy", env, device=device, n_steps=10, batch_size=48)
    model = model.load("policy.zip")

    imgs = []
    for i in range(args.max_episode_length):
        
        # PREDICT
        # obs_tmp = np.concatenate((obs["lowdim_ee"][:2], rod_pose, obs["lowdim_qpos"][:-1]))
        # actions, _state = model.predict(obs_tmp, deterministic=False)

        # ACT
        actions = np.random.uniform(-1, 1, size=env.action_shape)
        actions = np.array([0.5, 0.0, 0.0])
        next_obs, rew, done, _ = env.step(actions)
        imgs.append(env.render())
        obs = next_obs

        print("ee", obs["lowdim_ee"][:2])

    env.reset()

    imageio.mimsave("test_rollout.gif", np.stack(imgs), duration=3)
