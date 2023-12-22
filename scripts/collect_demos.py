import os
import time
import datetime
import imageio
import argparse
import numpy as np
import joblib

from tqdm import tqdm

from robot.robot_env import RobotEnv
from robot.controllers.oculus import VRController
from training.buffer import DictReplayBuffer

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str)
    parser.add_argument("--save_dir", type=str, default="data")
    # hardware
    parser.add_argument("--dof", type=int, default=6, choices=[3, 4, 6])
    parser.add_argument("--robot_type", type=str, default="panda", choices=["panda", "fr3"])
    parser.add_argument("--ip_address", type=str, default="172.16.0.1", choices=[None, "localhost", "172.16.0.1"])
    parser.add_argument("--camera_model", type=str, default="realsense", choices=["realsense", "zed"])
    # trajectories
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max_episode_length", type=int, default=100)

    args = parser.parse_args()

    args.exp = "test"
    assert args.exp is not None, "Specify --exp"
    env = RobotEnv(
        control_hz=10,
        DoF=args.dof,
        robot_type=args.robot_type,
        ip_address=args.ip_address,
        camera_model=args.camera_model,
        max_lin_vel=1.0,
        max_rot_vel=1.0,
        max_path_length=args.max_episode_length,
    )

    buffer_size = args.episodes * args.max_episode_length
    buffer = DictReplayBuffer(
        buffer_size,
        env.observation_shape,
        env.action_shape,
        obs_dict_type=env.observation_type,
        act_type=np.float32,
    )

    oculus = VRController()
    assert oculus.get_info()["controller_on"], "ERROR: oculus controller off"
    print("Oculus Connected")

    traj_idcs = []

    for i in range(args.episodes):

        env.reset()
        assert env._num_cameras > 0, "ERROR: camera(s) not connected!"
        print(f"Camera(s) Connected ({env._num_cameras})")
        
        print(f"Press 'A' to Start Collecting")
        # time to reset the scene
        while True:
            info = oculus.get_info()
            if info["success"]:
                # get obs after resetting scene
                obs = env.get_observation()
                break
        
        print(f"Press 'B' to Stop Collecting")

        start_pos = buffer.pos

        for j in tqdm(range(args.max_episode_length), desc=f"Collecting Trajectory {i}"):

            # wait for controller input
            info = oculus.get_info()
            while not (info["failure"] or info["movement_enabled"]):
                info = oculus.get_info()

            # press 'B' to end a trajectory
            if info["failure"]:
                continue

            # check if 'trigger' button is pressed
            if info["movement_enabled"]:

                # prepare obs for oculus
                pose = env._robot.get_ee_pose()
                gripper = env._robot.get_gripper_position()
                state = {"robot_state": {"cartesian_position": pose, "gripper_position": gripper}}
                action = oculus.forward(state)

                # prepare act
                if args.dof == 3:
                    action = np.concatenate((action[:3], action[-1:]))
                elif args.dof == 4:
                    action = np.concatenate((action[:3], action[5:6], action[-1:]))

                # step and record
                next_obs, rew, done, _ = env.step(action)
                
                buffer.push(obs, action, rew, next_obs, done)
                
                # env._robot.update_command(action, action_space="cartesian_velocity", blocking=False)
                # next_obs = env.get_observation()
                
                obs = next_obs
        
        print(f"Recorded Trajectory {i}")
        traj_idcs.append(buffer.pos)
        
    env.reset()

    print(f"Finished Collecting {i} Trajectories: save & exiting")

    save_dir = os.path.join(args.save_dir, args.exp)
    os.makedirs(save_dir, exist_ok=True)
    # save buffer
    joblib.dump(buffer, os.path.join(save_dir, "buffer.gz"), compress=3)
    # save traj indices
    joblib.dump(traj_idcs, os.path.join(save_dir, "idcs.gz"))

    print(f"Saved at {save_dir}")