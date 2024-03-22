import argparse
import datetime
import os
import time

import hydra
import imageio
import joblib
import numpy as np
import torch
from tqdm import tqdm, trange

from robot.controllers.motion_planner import MotionPlanner

# from robot.rlds_wrapper import (convert_rlds_to_np, load_rlds_dataset,
#                                 wrap_env_in_rlds_logger)
from robot.rlds_wrapper import DataCollectionWrapper
from robot.robot_env import RobotEnv
from utils.experiment import hydra_to_dict, set_random_seed, setup_wandb
from utils.transformations_mujoco import *


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

    for i in range(len(qpos_plan.ee_position)):

        des_pose = np.concatenate(
            (
                qpos_plan.ee_position[i].cpu().numpy(),
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

            curr_pose = env.unwrapped._robot.get_ee_pose()
            pos_diff = curr_pose[:3] - last_curr_pose[:3]
            angle_diff = curr_pose[3:] - last_curr_pose[3:]
            angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
            err = np.linalg.norm(pos_diff) + np.linalg.norm(angle_diff)

            # early stopping when actions don't change position anymore
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

    noise_std = 0. # 5e-2
    progress_threshold = 5e-2

    motion_planner = MotionPlanner(interpolation_dt=0.1, device=torch.device("cuda:0"))
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
    gripper = 0.0
    move_to_cartesian_pose(
        target_pose,
        gripper,
        motion_planner,
        controller,
        env,
        progress_threshold=progress_threshold,
    )

    target_pose[2] = 0.3 + np.random.normal(loc=0.0, scale=noise_std)
    gripper = 1.0
    move_to_cartesian_pose(
        target_pose,
        gripper,
        motion_planner,
        controller,
        env,
        progress_threshold=progress_threshold,
    )

    return env.get_obj_pose()[2] > 0.1


@hydra.main(
    config_path="../configs/", config_name="collect_cube_sim", version_base="1.1"
)
def run_experiment(cfg):

    logdir = os.path.join(cfg.log.dir, cfg.exp_id)
    os.makedirs(logdir, exist_ok=True)

    from asid.wrapper.asid_vec import make_env, make_vec_env

    cfg.robot.DoF = 6
    cfg.robot.gripper = True
    cfg.robot.on_screen_rendering = False
    cfg.robot.max_path_length = 100

    cfg.env.flatten = False
    cfg.robot.imgs = True

    cfg.env.obj_pose_noise_dict = None
    
    language_instruction = "pick up the red cube"

    robot_cfg_dict = hydra_to_dict(cfg.robot)
    robot_cfg_dict["blocking_control"] = True

    env = make_env(
        robot_cfg_dict=robot_cfg_dict,
        env_cfg_dict=hydra_to_dict(cfg.env),
        seed=cfg.seed,
        device_id=0,
    )

    env = DataCollectionWrapper(env, language_instruction=language_instruction, save_dir=f"data/{cfg.exp_id}/train")

    successes = []
    for i in trange(cfg.episodes):

        env.reset_buffer()

        success = collect_demo_pick_up(env)

        if success:
            env.save_buffer()

        successes.append(success)

        print(f"Recorded Trajectory {i}, success {success}")

    env.reset()

    print(f"Finished Collecting {i} Trajectories | Success {np.sum(successes)} / {len(successes)}")


if __name__ == "__main__":
    run_experiment()
