import os
import time
import datetime
import imageio
import argparse
import numpy as np
import joblib

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
    assert oculus.get_info()["controller_on"], "WARNING: oculus controller off"
    print("Oculus Connected")

    traj_idcs = []
    success_idcs = []

    for i in range(args.episodes):

        obs = env.reset()
        assert "img_obs_0" in obs.keys(), "ERROR: camera not connected!"
        
        print(f"Start Collecting Trajectory {i}")
        
        time_step = 0
        start_pos = buffer.pos

        while True and args.max_episode_length > time_step:
            
            # prepare obs for oculus
            pose = env._robot.get_ee_pose()
            gripper = env._robot.get_gripper_position()
            state = {"robot_state": {"cartesian_position": pose, "gripper_position": gripper}}
            action = oculus.forward(state)
            info = oculus.get_info()

            # press A or B to end a trajectory
            if info["success"] or info["failure"]:
                success = True if info["success"] else False
                success_idcs.append(success)
                break

            # check if tirgger button is pressed
            if info["movement_enabled"]:
                # prepare act
                if args.dof == 3:
                    action = np.concatenate((action[:3], action[-1:]))
                elif args.dof == 4:
                    action = np.concatenate((action[:3], action[5:6], action[-1:]))

                # step and record
                next_obs, rew, done, _ = env.step(action)
                
                buffer.push(obs, action, rew, next_obs, done)
                obs = next_obs
                print(f"Recorded Timestep {time_step} of Trajectory {i}")
                time_step += 1
        
        print(f"Recorded Trajectory {i}")
        traj_idcs.append(buffer.pos)
        
    env.reset()

    print("Finished Data Collection: save & exiting")

    save_dir = os.path.join(args.save_dir, args.exp)
    os.makedirs(save_dir, exist_ok=True)
    # save buffer
    joblib.dump(buffer, os.path.join(save_dir, "buffer.gz"), compress=3)
    # save traj indices
    joblib.dump(traj_idcs, os.path.join(save_dir, "idcs.gz"))
    joblib.dump(success_idcs, os.path.join(save_dir, "success.gz"))

    print(f"Saved at {save_dir}")