import argparse
import datetime
import os
import time

import hydra
import imageio
import joblib
import numpy as np
import torch
from tqdm import tqdm

from robot.controllers.motion_planner import MotionPlanner
from robot.rlds_wrapper import (convert_rlds_to_np, load_rlds_dataset,
                                wrap_env_in_rlds_logger)
from robot.robot_env import RobotEnv
from utils.experiment import hydra_to_dict, set_random_seed, setup_wandb
from utils.transformations_mujoco import *


class CartesianPDController:
    def __init__(self, Kp, Kd, control_hz=10):
        self.Kp = Kp  # Proportional gain
        self.Kd = Kd  # Derivative gain
        self.pos_prev_error = 0
        self.quat_prev_error = 0
        self.dt = 1/control_hz

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

def move_to_cartesian_pose(target_pose, gripper, motion_planner, controller, env, progress_threshold=1e-3, max_iter_per_waypoint=20, render=False, verbose=False):
    
    controller.reset()

    # start = env.unwrapped._robot.get_joint_positions().copy()
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
    error = []
    imgs = []

    for i in range(len(qpos_plan.ee_position)):

        des_pose = np.concatenate((qpos_plan.ee_position[i].cpu().numpy(), quat_to_euler_mujoco(qpos_plan.ee_quaternion[i].cpu().numpy())))
        # des_pose[5] = des_pose[5] / 2 # scale to np.pi/2
        
        print("des_pose", des_pose)
        last_curr_pose = des_pose

        for j in range(max_iter_per_waypoint):
            
            # get current pose
            curr_pose = env.unwrapped._robot.get_ee_pose()

            # run PD controller
            act = controller.update(curr_pose, des_pose)
            # act[3:] = np.arctan2(np.sin(act[3:] * 2), np.cos(act[3:] * 2)) / 2
            act = np.concatenate((act, [gripper]))

            print("angle act", act[3:], "euler", curr_pose[3:])
            # print("angle act", act[3:], "euler", curr_pose[3:])
            # step env
            obs, _, _, _ = env.step(act)
            steps += 1
            
            # compute error
            if verbose:
                curr_pose = env.unwrapped._robot.get_ee_pose()
                err_pos = np.linalg.norm(target_pose[:3]-curr_pose[:3])
                err_angle = np.linalg.norm(target_pose[3:] - curr_pose[3:])
                err = err_pos # + err_angle
                error.append(err)

                # print(j, "err", err_pos, err_angle, "pose norm", np.linalg.norm(last_curr_pose-curr_pose)) # "act_max_abs", np.max(np.abs(act)), "act", act)

            if render:
                imgs.append(env.render())
            env.render()

            # early stopping when actions don't change position anymore
            if np.linalg.norm(target_pose[:3]-curr_pose[:3]) < progress_threshold:
                break
            last_curr_pose = curr_pose

    return imgs, steps


def collect_demo_pick_up(
    env, render=False, verbose=False,
):
    """
    Collect a "pick up the red block" demo

    Args:
    - env: robot environment
    - z_waypoints: list of z waypoints for the pick up -> above, closer, down
    - noise_std: list of noise std for each waypoint
    - render: whether to render the environment

    Returns:
    - success: whether the pick up was successful
    - imgs: list of images for each action (empty if render=False)
    """

    noise_std = 0 # 5e-2

    motion_planner = MotionPlanner(interpolation_dt=0.5, device=torch.device("cuda:0"))
    controller = CartesianPDController(Kp=0.7, Kd=0., control_hz=10)
    imgs = []

    progress_threshold = 0.3

    env.reset()

    # get initial target pose
    target_pos = env.get_obj_pose().copy()[:3]
    target_quat = quat_to_euler_mujoco(env.get_obj_pose().copy()[3:])
    target_pose = np.concatenate((target_pos, target_quat))

    curr_pose = env.unwrapped._robot.get_ee_pose()
    # angular_diff = np.arctan2(np.sin(target_pose[3:] - curr_pose[3:]), np.cos(target_pose[3:] - curr_pose[3:]))
    # target_pose[3:] = curr_pose[3:] + angular_diff

    # overwrite x,y angle w/ gripper default
    target_pose[3:5] = env.unwrapped._default_angle[:2]

    # target_pose[3:5] = env.unwrapped._default_angle[:3]
    # target_pose[5] -= np.pi / 4
    # target_pose[5] = np.clip(target_pose[5], -np.pi/2, np.pi/2)

    # randomize grasp angle z
    rng = np.random.randint(0, 3)
    target_pose[5:] += np.pi / 2 if rng else -np.pi / 2 if rng == 1 else 0
    # target_pose[3:] = np.arctan2(np.sin(target_pose[3:]), np.cos(target_pose[3:]))

    # WARNING: real robot EE is offset by 90 deg -> target_pose[5] += np.pi / 4

    target_pose[2] = 0.3 + np.random.normal(loc=0.0, scale=noise_std)
    gripper = 0.0
    tmp, _ = move_to_cartesian_pose(target_pose, gripper, motion_planner, controller, env, progress_threshold=progress_threshold, max_iter_per_waypoint=20, render=render, verbose=True)
    imgs += tmp

    target_pose[2] = 0.12 + np.random.normal(loc=0.0, scale=noise_std/2)
    gripper = 0.0
    tmp, _ = move_to_cartesian_pose(target_pose, gripper, motion_planner, controller, env, progress_threshold=progress_threshold, max_iter_per_waypoint=20, render=render, verbose=True)
    imgs += tmp

    # target_pose[2] = 0.2 + np.random.normal(loc=0.0, scale=noise_std)
    # gripper = 1.0
    # tmp, _ = move_to_cartesian_pose(target_pose, gripper, motion_planner, controller, env, progress_threshold=progress_threshold, max_iter_per_waypoint=20, render=render, verbose=True)
    # imgs += tmp

    target_pose[2] = 0.3 + np.random.normal(loc=0.0, scale=noise_std)
    gripper = 1.0
    tmp, _ = move_to_cartesian_pose(target_pose, gripper, motion_planner, controller, env, progress_threshold=progress_threshold, max_iter_per_waypoint=20, render=render, verbose=True)
    imgs += tmp


    success = env.get_obj_pose()[2] > 0.1
    return success, imgs


@hydra.main(
    config_path="../configs/", config_name="collect_cube_sim", version_base="1.1"
)
def run_experiment(cfg):

    logdir = os.path.join(cfg.log.dir, cfg.exp_id, str(cfg.seed), "sim")
    os.makedirs(logdir, exist_ok=True)

    from asid.wrapper.asid_vec import make_env, make_vec_env
    
    cfg.robot.DoF = 6
    cfg.robot.gripper = True
    cfg.robot.on_screen_rendering = True
    cfg.robot.max_path_length = 100

    cfg.env.flatten = False
    cfg.env.obj_pos_noise = True

    env = make_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        env_cfg_dict=hydra_to_dict(cfg.env),
        seed=cfg.seed,
        device_id=0,
    )

    # with wrap_env_in_rlds_logger(
    #     env, cfg.exp_id, logdir, max_episodes_per_shard=1
    # ) as rlds_env:
    for i in range(cfg.episodes):

        obss = []
        acts = []

        # obs = rlds_env.reset()

        success, _ = collect_demo_pick_up(
            env,
            # rlds_env,
            render=True,
        )

        print(f"Recorded Trajectory {i}, success {success}")

    env.reset()

    # check if dataset was saved
    loaded_dataset = load_rlds_dataset(logdir)

    print(f"Finished Collecting {i} Trajectories")


if __name__ == "__main__":
    run_experiment()
