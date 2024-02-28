import argparse
import datetime
import os
import time
import hydra
import imageio
import joblib
import numpy as np
import torch

from perception.trackers.color_tracker import ColorTracker
from robot.robot_env import RobotEnv
from utils.pointclouds import crop_points
from utils.transformations import euler_to_rmat, quat_to_euler
from utils.transformations_mujoco import *

from asid.wrapper.asid_vec import make_env, make_vec_env
from utils.experiment import hydra_to_dict, set_random_seed, setup_wandb

from asid.scripts.collect_demos_sim_curobo import *

def move_to_target(env, motion_planner, target_pose):

    start = env.unwrapped._robot.get_ee_pose()
    start = np.concatenate((start[:3], euler_to_quat_mujoco(start[3:])))

    # IMPORTANT: flip yaw angle mujoco to curobo!
    if env.unwrapped.sim:
        start[5] = -start[5]
    goal = np.concatenate((target_pose[:3], euler_to_quat_mujoco(target_pose[3:6])))
    qpos_plan = motion_planner.plan_motion(start, goal, return_ee_pose=False)
    for i in range(len(qpos_plan)):
        # real -> but slow
        # env.unwrapped._robot.update_joints(qpos_plan[i].position.cpu().numpy().tolist(), velocity=False, blocking=True)
        if env.unwrapped.sim:
            env.unwrapped._robot.move_to_joint_positions(qpos_plan[i].position.cpu().numpy())
        # env.unwrapped._robot.update_desired_joint_positions(qpos_plan[i].position.cpu().tolist())
        # time.sleep(0.1)
        env.render()
        
@hydra.main(
    config_path="configs/", config_name="collect_rod_real", version_base="1.1"
)
def run_experiment(cfg):

    # real
    # cfg.robot.calibration_file = "perception/cameras/calibration/logs/aruco/24_02_27_16_28_52.json"
    # cfg.robot.camera_model = "realsense"

    # sim
    # cfg.robot.on_screen_rendering = False
    # cfg.env.obj_pos_noise = True
    
    cfg.robot.gripper = True

    env = make_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        env_cfg_dict=hydra_to_dict(cfg.env),
        seed=cfg.seed,
        device_id=0,
        verbose=False,
    )

    tracker = ColorTracker(outlier_removal=False)
    tracker.reset()
    # define workspace
    crop_min = [0.0, -0.6, -0.1]
    crop_max = [0.7, 0.6, 0.5]

    obs = env.reset()

    imgs = []
    actions = np.ones(env.action_shape)

    # run for 25 steps to eliminate realsense noise or get filtered estimate
    for i in range(25):
        obs_dict = env.get_images_and_points()
        rgbs, points = [], []
        for key in obs_dict.keys():
            rgbs.append(obs_dict[key]["rgb"])
            points.append(obs_dict[key]["points"])
        tracked_points = tracker.track_multiview(rgbs, points, color="red", show=False)
        cropped_points = crop_points(
            tracked_points, crop_min=crop_min, crop_max=crop_max
        )

        rod_pose = tracker.get_rod_pose(
            cropped_points,
            lowpass_filter=False,
            cutoff_freq=1,
            control_hz=cfg.robot.control_hz,
            show=False,
        )
        time.sleep(0.1)

    motion_planner = MotionPlanner(interpolation_dt=0.3, device=torch.device("cuda:0"))

    env.reset()
    
    # get rod pose
    rod_pose = env.get_obj_pose()
    target_pos = rod_pose[:3]

    # convert euler to mujoco quat
    if env.unwrapped.sim:
        target_orn = quat_to_euler_mujoco(rod_pose[3:])
    else:
        target_orn = quat_to_euler(rod_pose[3:])
    init_rod_pitch = target_orn[0]
    init_rod_yaw = target_orn[2]

    # up right ee
    target_orn[:1] = env.unwrapped._default_angle[:1]

    # align pose -> grasp pose
    target_orn[2] += np.pi / 2 

    # IMPORTANT: flip yaw angle mujoco to curobo!
    target_orn[2] = -target_orn[2]

    # gripper is symmetric
    if target_orn[2] > np.pi / 2:
        target_orn[2] -= np.pi
    if target_orn[2] < -np.pi / 2:
        target_orn[2] += np.pi

    # Set grasp target to center of mass
    com = -0.1
    target_pos[0] -= com * np.sin(init_rod_yaw)
    target_pos[1] += com * np.cos(init_rod_yaw)


    # MOVE ABOVE
    target_pos[2] += 0.2
    move_to_target(env, motion_planner, np.concatenate((target_pos, target_orn)))

    # MOVE DOWN
    target_pos[2] = 0.17
    move_to_target(env, motion_planner, np.concatenate((target_pos, target_orn)))

    # MOVE UP
    target_pos[2] = 0.3
    move_to_target(env, motion_planner, np.concatenate((target_pos, target_orn)))

    # PICK
    # env.unwrapped._robot.update_gripper(1.0, velocity=False, blocking=True)
    # imgs.append(env.render())
    move_to_target(env, motion_planner, target_pose)

    # MOVE UP
    target_pose[2] = 0.3
    # env.unwrapped._robot.update_pose(target_pose, blocking=True)
    # imgs.append(env.render())
    move_to_target(env, motion_planner, target_pose)

    # # MOVE TO YELLOW BLOCK
    # target_pose = torch.tensor(
    #     [0.28793925, -0.35449123, 0.12003776, 3.07100117, 0.01574947, -0.80526758]
    #     # [0.28867856, -0.40134683, 0.11756707, 3.13773595, 0.0078624, -0.70369389]
    # )
    # target_pose[2] = 0.3
    # env.unwrapped._robot.update_pose(target_pose, blocking=True)
    # imgs.append(env.render())

    # # MOVE DOWN + yellow block height + margin
    # target_pose[2] = 0.12 + 0.05 + 0.02
    # env.unwrapped._robot.update_pose(target_pose, blocking=True)
    # imgs.append(env.render())

    # # DROP
    # env.unwrapped._robot.update_gripper(0.0, velocity=False, blocking=True)
    # imgs.append(env.render())

    # # MOVE UP
    # target_pose[2] = 0.3
    # env.unwrapped._robot.update_pose(target_pose, blocking=True)
    # imgs.append(env.render())

    env.reset()

    # imageio.mimsave("test_rollout.gif", np.stack(imgs), duration=3)


if __name__ == "__main__":
    run_experiment()
