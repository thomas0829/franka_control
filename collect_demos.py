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
from robot.rlds_wrapper import wrap_env_in_rlds_logger, load_rlds_dataset, convert_rlds_to_np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str)
    parser.add_argument("--save_dir", type=str, default="data")
    # hardware
    parser.add_argument("--dof", type=int, default=6, choices=[3, 4, 6])
    parser.add_argument(
        "--robot_type", type=str, default="panda", choices=["panda", "fr3"]
    )
    parser.add_argument(
        "--ip_address",
        type=str,
        default="172.16.0.1",
        choices=[None, "localhost", "172.16.0.1"],
    )
    parser.add_argument(
        "--camera_model", type=str, default="realsense", choices=["realsense", "zed"]
    )
    # trajectories
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max_episode_length", type=int, default=1000)

    args = parser.parse_args()

    assert args.exp is not None, "Specify --exp"
    save_dir = os.path.join(args.save_dir, args.exp)
    os.makedirs(save_dir, exist_ok=True)

    env = RobotEnv(
        control_hz=10,
        DoF=args.dof,
        robot_type=args.robot_type,
        ip_address=args.ip_address,
        camera_model=args.camera_model,
        max_path_length=args.max_episode_length,
    )

    oculus = VRController()
    assert oculus.get_info()["controller_on"], "ERROR: oculus controller off"
    print("Oculus Connected")

    with wrap_env_in_rlds_logger(env, args.exp, save_dir, max_episodes_per_shard=1) as rlds_env:
        for i in range(args.episodes):
            rlds_env.reset()
            # assert env._num_cameras > 0, "ERROR: camera(s) not connected!"
            print(f"Camera(s) Connected ({rlds_env.unwrapped._num_cameras})")

            print(f"Press 'A' to Start Collecting")
            # time to reset the scene
            while True:
                info = oculus.get_info()
                if info["success"]:
                    # get obs after resetting scene
                    obs = rlds_env.unwrapped.get_observation()
                    break

            print(f"Press 'B' to Stop Collecting")

            obss = []
            acts = []

            for j in tqdm(
                range(args.max_episode_length), desc=f"Collecting Trajectory {i}"
            ):
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
                    pose = rlds_env.unwrapped._robot.get_ee_pose()
                    gripper = rlds_env.unwrapped._robot.get_gripper_position()
                    state = {
                        "robot_state": {
                            # "cartesian_position": np.concatenate(pose),
                            "cartesian_position": pose,
                            "gripper_position": gripper,
                        }
                    }
                    vel_act = oculus.forward(state)

                    # convert vel to delta actions
                    delta_act = rlds_env.unwrapped._robot._ik_solver.cartesian_velocity_to_delta(vel_act)
                    delta_gripper = rlds_env.unwrapped._robot._ik_solver.gripper_velocity_to_delta(
                        vel_act[-1:]
                    )
                    # prepare act
                    if args.dof == 3:
                        act = np.concatenate((delta_act[:3], delta_act[-1:]))
                    elif args.dof == 4:
                        act = np.concatenate((delta_act[:3], delta_act[5:6], vel_act[-1:]))
                    elif args.dof == 6:
                        act = np.concatenate((delta_act, vel_act[-1:]))

                    next_obs, rew, done, _ = rlds_env.step(rlds_env.type_action(act))

                    obss.append(obs)
                    acts.append(act)

                    obs = next_obs

            print(f"Recorded Trajectory {i}")

    env.reset()

    # check if dataset was saved
    loaded_dataset = load_rlds_dataset(save_dir)

    print(f"Finished Collecting {i} Trajectories")