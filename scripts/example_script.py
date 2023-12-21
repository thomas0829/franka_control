import imageio
import argparse
import torch
import numpy as np
import joblib

from robot.robot_env import RobotEnv

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    # experiment
    parser.add_argument("--exp", type=str)
    parser.add_argument("--save_dir", type=str, default="data")
    
    # hardware
    parser.add_argument("--dof", type=int, default=6, choices=[3, 4, 6])
    parser.add_argument("--robot_type", type=str, default="panda", choices=["panda", "fr3"])
    parser.add_argument("--ip_address", type=str, default="172.16.0.1", choices=[None, "localhost", "172.16.0.1"])
    parser.add_argument("--camera_model", type=str, default="realsense", choices=["realsense", "zed"])
    
    # training
    parser.add_argument("--max_episode_length", type=int, default=10)
    parser.add_argument("--gpu_id", type=int, default=0)

    args = parser.parse_args()

    assert args.exp is not None, "Specify --exp"
    device = torch.device(("cuda:" + str(args.gpu_id)) if args.gpu_id >= 0. and torch.cuda.is_available() else "cpu")
    
    env = RobotEnv(
        control_hz=10,
        DoF=args.dof,
        robot_type=args.robot_type,
        gripper=True,
        ip_address=args.ip_address,
        camera_model=args.camera_model,
        max_lin_vel=1.0,
        max_rot_vel=1.0,
        max_path_length=args.max_episode_length,
    )

    obs = env.reset()

    imgs = []
    for i in range(args.max_episode_length):
        
        action = np.random.uniform(-1, 1, size=env.action_shape)
        next_obs, rew, done, _ = env.step(action)
        imgs.append(env.render())
        obs = next_obs

    imageio.mimsave("test_rollout.gif", np.stack(imgs), duration=3)
