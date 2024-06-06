import argparse
import datetime
import os
import random
import time

import hydra
import imageio
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm, trange

from robot.controllers.motion_planner import MotionPlanner
from robot.sim.vec_env.vec_env import make_env
from robot.wrappers.crop_wrapper import CropImageWrapper
from robot.wrappers.data_wrapper import DataCollectionWrapper
from robot.wrappers.resize_wrapper import ResizeImageWrapper
from utils.experiment import hydra_to_dict, set_random_seed, setup_wandb
from utils.transformations_mujoco import *
from robot.sim.mujoco.distractor_wrapper import DistWrapper

import logging
logging.getLogger('curobo').setLevel(logging.WARNING)

class CartesianPDController:
    def __init__(self, Kp, Kd, control_hz=10):
        self.Kp = Kp  # Proportional gain
        self.Kd = Kd  # Derivative gain
        self.pos_prev_error = 0
        self.quat_prev_error = 0
        self.dt = 1 / control_hz

    def reset(self):
        self.pos_prev_error = 0
        self.quat_prev_error = 0

    def update(self, curr, des):
        """
        Update the PD controller.

        Args:
            des (float): The desired value.
            curr (float): The current value.

        Returns:
            float: The control output.
        """
        # Calculate the position error
        pos_error = des[:3] - curr[:3]

        # Calculate the derivative of the position error
        pos_error_dot = (pos_error - self.pos_prev_error) / self.dt

        # Update the previous position error and time for the next iteration
        self.pos_prev_error = pos_error

        # Calculate the position control output
        u_pos = self.Kp * pos_error + self.Kd * pos_error_dot

        # Calculate the quaternion error
        # quat_error = subtract_euler_mujoco(des[3:], curr[3:])
        quat_error = des[3:] - curr[3:]
        quat_error = np.arctan2(np.sin(quat_error), np.cos(quat_error))

        # Calculate the derivative of the quaternion error
        quat_error_dot = (quat_error - self.quat_prev_error) / self.dt

        # Update the previous quaternion error and time for the next iteration
        self.quat_prev_error = quat_error

        # Calculate the quaternion control output
        u_quat = self.Kp * quat_error + self.Kd * quat_error_dot

        # Combine the position and quaternion control outputs
        u = np.concatenate((u_pos, u_quat))

        return u


def move_to_cartesian_pose(
    target_pose,
    gripper,
    motion_planner,
    controller,
    env,
    progress_threshold=1e-3,
    max_iter_per_waypoint=20,
):

    controller.reset()

    start = env.unwrapped._robot.get_ee_pose().copy()
    start = np.concatenate((start[:3], euler_to_quat_mujoco(start[3:])))
    target_pose = target_pose.copy()

    if target_pose[5] > np.pi / 2:
        target_pose[5] -= np.pi
    if target_pose[5] < -np.pi / 2:
        target_pose[5] += np.pi

    goal = np.concatenate((target_pose[:3], euler_to_quat_mujoco(target_pose[3:])))
    qpos_plan = motion_planner.plan_motion(start, goal, return_ee_pose=True)

    steps = 0
    imgs = []

    # first waypoint is current pose -> start from i=1
    for i in range(len(qpos_plan.ee_position)-1):
        des_pose = np.concatenate(
            (
                qpos_plan.ee_position[i+1].cpu().numpy(),
                quat_to_euler_mujoco(qpos_plan.ee_quaternion[i].cpu().numpy()),
            )
        )
        last_curr_pose = env.unwrapped._robot.get_ee_pose()

        for j in range(max_iter_per_waypoint):

            # get current pose
            curr_pose = env.unwrapped._robot.get_ee_pose()

            # run PD controller
            act = controller.update(curr_pose, des_pose)
            act = np.concatenate((act, [gripper]))

            # step env
            obs, _, _, _ = env.step(act)
            steps += 1
         

            image = obs[f"front_rgb"]
    
            # import cv2
            # cv2.imshow('Real-time video', cv2.cvtColor(image,
            #                                            cv2.COLOR_BGR2RGB))

            # # Press 'q' on the keyboard to exit the loop
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            curr_pose = env.unwrapped._robot.get_ee_pose()
            pos_diff = curr_pose[:3] - last_curr_pose[:3]
            angle_diff = curr_pose[3:] - last_curr_pose[3:]
            angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
            err = np.linalg.norm(pos_diff) + np.linalg.norm(angle_diff)

            # early stopping when actions don't change position anymore
            # 5x more accuracy for last 3 steps
            # if i > len(qpos_plan.ee_position) - 3:
            #     if err < progress_threshold / 5:
            #         break
            # elif err < progress_threshold:
            #     break
            if err < progress_threshold:
                break

            last_curr_pose = curr_pose

    return imgs, steps


def collect_demo_pick_up(env):
    """
    Collect a "pick up the red block" demo

    Args:
    - env: robot environment
    - z_waypoints: list of z waypoints for the pick up -> above, closer, down
    - noise_std: list of noise std for each waypoint

    Returns:
    - success: whether the pick up was successful
    """

    target_noise_std = 0.0
    progress_threshold = 3e-2

    motion_planner = MotionPlanner(
        interpolation_dt=0.1, random_obstacle=False, device=torch.device("cuda:0")
    )
    controller = CartesianPDController(
        Kp=1.0, Kd=0.0, control_hz=env.unwrapped._robot.control_hz
    )

    env.reset()

    # get initial target pose
    target_pos = env.get_obj_pose().copy()[:3]
    target_quat = quat_to_euler_mujoco(env.get_obj_pose().copy()[3:])
    target_pose = np.concatenate((target_pos, target_quat))

    # overwrite x,y angle w/ gripper default
    target_pose[3:5] = env.unwrapped._default_angle[:2]

    # WARNING: real robot EE is offset by 90 deg -> target_pose[5] += np.pi / 4

    # lowest possible z w/ Curobo | real is 0.13
    target_pose[2] = 0.12
    target_pose_noisy = target_pose + np.random.normal(
        loc=0.0, scale=target_noise_std, size=target_pose.shape
    )
    gripper = 0.0
    move_to_cartesian_pose(
        target_pose_noisy,
        gripper,
        motion_planner,
        controller,
        env,
        progress_threshold=progress_threshold,
    )

    target_pose[2] = 0.2
    # randomize lift up position
    target_pose_noisy = target_pose + np.random.normal(
        loc=0.0, scale=target_noise_std, size=target_pose.shape
    )
    gripper = 1.0
    move_to_cartesian_pose(
        target_pose_noisy,
        gripper,
        motion_planner,
        controller,
        env,
        progress_threshold=progress_threshold,
    )

    return env.get_obj_pose()[2] > 0.05


@hydra.main(
    config_path="../../configs/", config_name="collect_cube_sim", version_base="1.1"
)
def run_experiment(cfg):

    logdir = os.path.join(cfg.log.dir, cfg.exp_id)
    os.makedirs(logdir, exist_ok=True)

    cfg.robot.max_path_length = cfg.max_episode_length

    # env = make_env(
    #     robot_cfg_dict=hydra_to_dict(cfg.robot),
    #     env_cfg_dict=hydra_to_dict(cfg.env),
    #     seed=cfg.seed,
    #     device_id=0,
    #     verbose=True,
    # )

    def make_multi_cube_env(cfg, color_ids=[0, 1, 2], savedir=None):

        colors = [
            [0., 0.9, 0., 1.],
            [0.9, 0., 0., 1.],
            [0., 0., 0.9, 1.]
        ]
        color_names = [
            "green",
            "red",
            "blue"
        ]
        offsets = [0.1, -0.1]

        language_instruction = f"pick up the {color_names[color_ids[0]]} cube"

        robot_cfg_dict = hydra_to_dict(cfg.robot)
        robot_cfg_dict["model_name"] = "two_cubes_franka"

        env_cfg_dict = hydra_to_dict(cfg.env)
        env_cfg_dict["obj_id"] = "cube"
        env_cfg_dict["obj_rgba"] = colors[color_ids[0]]
        # env_cfg_dict["obj_pose_init"][1] = offsets[color_ids[0]]
        # env_cfg_dict["obj_pose_noise_dict"] = None
        # env_cfg_dict["obj_pose_noise_dict"] = {
        #     "x": { "min": -0.1, "max": 0.1 },
        #     "y": { "min": -0.1, "max": 0.1 },
        #     # "yaw": { "min": -0., "max": 0.0 },
        #     "yaw": { "min": -0.785, "max": 0.785 },
        # }
        env = make_env(
                robot_cfg_dict=robot_cfg_dict,
                env_cfg_dict=env_cfg_dict,
                seed=cfg.seed,
                device_id=0,
                verbose=True,
            )

        dis_cfg_dict = hydra_to_dict(cfg.env)
        dis_cfg_dict["reset_data_on_reset"] = False
        dis_cfg_dict["obj_id"] = "distractor_0"
        dis_cfg_dict["obj_rgba"] = colors[color_ids[1]]
        # dis_cfg_dict["obj_pose_init"][1] = offsets[color_ids[1]]
        # dis_cfg_dict["obj_pose_noise_dict"] = None
        # env_cfg_dict["obj_pose_noise_dict"] = {
        #     "x": { "min": -0.1, "max": 0.1 },
        #     "y": { "min": -0.1, "max": 0.1 },
        #     # "yaw": { "min": -0., "max": 0.0 },
        #     "yaw": { "min": -0.785, "max": 0.785 },
        # }
        env = DistWrapper(env, **dis_cfg_dict)

        env = DataCollectionWrapper(
            env,
            language_instruction=language_instruction,
            fake_blocking=False,
            act_noise_std=cfg.act_noise_std,
            save_dir=savedir,
        )

        return env


    for split, n_episodes in zip(
        ["train", "eval"], [cfg.episodes, int(cfg.episodes // 10)]
        # ["eval"], [int(cfg.episodes // 10)]
    ):

        if cfg.exp_id[0] == "/":
            savedir = f"{cfg.exp_id}/{split}"
        else:
            savedir = f"data/{cfg.exp_id}/{split}"
        
        successes = []
        obj_poses = []

        n_traj = 0

        # for n_traj in trange(cfg.episodes):
        with tqdm(total=n_episodes) as pbar:
            while n_traj < n_episodes:
            
                # color #0 for first half of demos
                if n_traj == 0:
                    env = make_multi_cube_env(cfg, color_ids=[0, 1, 2], savedir=savedir)
                # color #1 for second half of demos
                elif n_traj == n_episodes // 2:
                    traj_count_prev = env.traj_count
                    env = make_multi_cube_env(cfg, color_ids=[1, 0, 2], savedir=savedir)
                    env.traj_count = traj_count_prev
                
                env.reset_buffer()

                try:
                    success = collect_demo_pick_up(env)
                    if success:
                        print(f"SUCCESS")
                        env.save_buffer()
                        n_traj += 1
                        pbar.update(1)
                        pbar.set_postfix_str(f'{n_traj}/{n_episodes}')
                    else:
                        print(f"FAILURE")

                    tmp_pose = env.buffer[0]["obj_pose"].copy()
                    obj_poses.append(
                        np.concatenate((tmp_pose[:3], quat_to_euler_mujoco(tmp_pose[3:])))
                    )
                    successes.append(success)

                # catch Curobo ValueError
                except ValueError as e:
                    success = False
                    print(e)

        obj_poses = np.stack(obj_poses)
        successes = np.array(successes)

        # dump statistics
        np.save(
            os.path.join(savedir, f"obj_poses_{split}"),
            {"obj_poses": obj_poses, "successes": successes},
        )

        # plot position stats
        poss = [pos for pos in obj_poses[:, :3]]
        poss = np.stack(poss)
        plt.scatter(
            poss[successes, 1], poss[successes, 0], color="tab:blue", label="success"
        )
        plt.scatter(
            poss[~successes, 1],
            poss[~successes, 0],
            color="tab:orange",
            marker="X",
            label="failure",
        )
        plt.legend()
        plt.xlabel("y")
        plt.ylabel("x")
        plt.savefig(os.path.join(savedir, f"obj_pos_{split}.png"))
        plt.close()

        # plot orientation stats
        oris = [ori for ori in obj_poses[:, 3:]]
        oris = np.stack(oris)
        plt.figure(figsize=(20, 2))
        plt.scatter(
            oris[successes, 2],
            np.zeros_like(oris[successes, 0]),
            color="tab:blue",
            label="success",
        )
        plt.scatter(
            oris[~successes, 2],
            np.zeros_like(oris[~successes, 0]),
            color="tab:orange",
            marker="X",
            label="failure",
        )
        plt.legend()
        plt.xlabel("yaw")
        plt.savefig(os.path.join(savedir, f"obj_ori_{split}.png"))
        plt.close()

        env.reset()

        print(
            f"Finished Collecting {n_traj} Trajectories | Success {np.sum(successes)} / {len(successes)} | {split}"
        )


if __name__ == "__main__":
    run_experiment()
