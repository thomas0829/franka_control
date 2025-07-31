import os
import time

import hydra
import numpy as np
from tqdm import tqdm

from robot.controllers.oculus import VRController
from robot.wrappers.crop_wrapper import CropImageWrapper
from robot.wrappers.data_wrapper import DataCollectionWrapper
from robot.wrappers.resize_wrapper import ResizeImageWrapper
from robot.sim.vec_env.vec_env import make_env
from utils.experiment import hydra_to_dict
import cv2
import pickle

from absl import flags
FLAGS = flags.FLAGS
import sys
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import date

# Setup logger
import logging
from datetime import date

# Setup logger
logger = logging.getLogger("collect_demos")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

class LogColors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'

def log_config(msg):
    logger.info(f"{LogColors.CYAN}{msg}{LogColors.END}")

def log_connect(msg):
    logger.info(f"{LogColors.BLUE}{msg}{LogColors.END}")

def log_instruction(msg):
    logger.info(f"{LogColors.YELLOW}{msg}{LogColors.END}")

def log_success(msg):
    logger.info(f"{LogColors.GREEN}{msg}{LogColors.END}")

def log_failure(msg):
    logger.info(f"{LogColors.RED}{msg}{LogColors.END}")

def log_important(msg):
    logger.info(f"{LogColors.BOLD}{LogColors.HEADER}{msg}{LogColors.END}")

# Initialize the 3D plot
def initialize_3d_plot():
    plt.ion()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("3D Visualization of XYZ Position")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.grid()

    range = [-0.05, 0.05]
    # Set fixed axis ranges
    ax.set_xlim(range)  # Adjust as needed
    ax.set_ylim(range)  # Adjust as needed
    ax.set_zlim(range)   # Adjust as needed

    return fig, ax

# Update the 3D plot with new data
def update_3d_plot(ax, xyz, target_offset=None, robot_offset=None):
    ax.clear()
    ax.set_title("3D Visualization of XYZ Position")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.grid()

    range = [-0.05, 0.05]
    # Set fixed axis ranges
    ax.set_xlim(range)  # Adjust as needed
    ax.set_ylim(range)  # Adjust as needed
    ax.set_zlim(range)   # Adjust as needed
    # Plot the position as a vector from the origin
    ax.quiver(
        0, 0, 0,  # Origin
        xyz[0], xyz[1], xyz[2],  # Vector components
        length=1.0, color='r', label="Position Vector (XYZ)"
    )

    if target_offset is not None:
        ax.quiver(
            0, 0, 0,  # Origin
            target_offset[0], target_offset[1], target_offset[2],  # Vector components
            length=1.0, color='g', label="Target Pose Offset"
        )

    if robot_offset is not None:
        ax.quiver(
            0, 0, 0,  # Origin
            robot_offset[0], robot_offset[1], robot_offset[2],  # Vector components
            length=1.0, color='b', label="Robot Pose Offset"
        )
    
    ax.legend()
    plt.pause(0.01)  # Pause to update the plot

def get_input_action(env, oculus, cfg):
    """
    Get action from oculus controller
    """
    # prepare obs for oculus
    pose = env.unwrapped._robot.get_ee_pose()
    gripper = env.unwrapped._robot.get_gripper_position()
    state = {
        "robot_state": {
            "cartesian_position": pose,
            "gripper_position": gripper,
        }
    }

    vel_act, info =  oculus.forward(state, include_info=True, method="delta_action")
    # update_3d_plot(ax, info["delta_action"][:3], info["target_pos_offset"][:3], info["robot_pos_offset"][:3])
    
    # convert vel to delta actions
    delta_act = env.unwrapped._robot._ik_solver.cartesian_velocity_to_delta(
        vel_act
    )

    # prepare act
    if cfg.robot.DoF == 3:
        act = np.concatenate((delta_act[:3], vel_act[-1:]))
    elif cfg.robot.DoF == 4:
        act = np.concatenate((delta_act[:3], delta_act[5:6], vel_act[-1:]))
    elif cfg.robot.DoF == 6:
        act = np.concatenate((delta_act, vel_act[-1:]))
        
    if oculus.vr_state["gripper"] > 0.5:
        act[-1] = 0.5
    else:
        act[-1] = 0
    
    return act


@hydra.main(
    # config_path="../../configs/", config_name="collect_demos_real", version_base="1.1"
    config_path="../../configs/", config_name="sim_to_real_config", version_base="1.1"
)
def run_experiment(cfg):
    FLAGS(sys.argv)
    cfg.robot.max_path_length = cfg.max_episode_length
    assert cfg.robot.imgs, "ERROR: set robot.imgs=true to record image observations!"

    # configs
    log_config(f"language instruction: {cfg.language_instruction}")
    log_config(f"number of episodes: {cfg.episodes}")
    log_config(f"control hz: {cfg.robot.control_hz}")
    log_config(f"dataset name: {cfg.exp_id}")
    savedir = f"{cfg.base_dir}/date_{date.today().month}{date.today().day}/npy/{cfg.exp_id}/{cfg.split}"
    log_config(f"save directory: {savedir}")

    # No-ops related variables
    no_ops_threshold = cfg.no_ops_threshold
    mode = cfg.mode
    no_ops_last_detected_time = cfg.no_ops_last_detected_time
    log_config(f"no_ops_threshold: {no_ops_threshold} secs")
    log_config(f"data collection mode: {mode}")
    
    # initialize env
    log_connect("Initializing env...")
    env = make_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        seed=cfg.seed,
        device_id=0,
        verbose=True,
    )

    debug = False
    gt_joint_poses = np.load(cfg.gt_joint_pos_filepath)
    number_times = 5

    if debug:
        gt_joint_poses = gt_joint_poses[:1, :]
        number_times = 1
        # breakpoint()


    save_list = [] # each element is pair of (gt_joint_pos, actual_joint_pos)
    for i, gt_joint_pos in enumerate(gt_joint_poses):
        obs = env.reset()
    
        log_success("Resetting env")
        log_important(f"[{i}] gt joint pos: {gt_joint_pos}")

        # 10 trials
        count = 0

        while count != number_times:
            log_important(f"trial {count}")
            start = input("start?(y/n)")
            while start != "y" and start != "n":
                start = input("start?(y/n)")
            
            if start:
                for _ in range(1):
                    env._robot.update_joints(
                        gt_joint_pos.tolist(), velocity=False, blocking=True
                    )

                time.sleep(0.5)
                log_success("finished")
                current_joint_pos = env._robot.get_joint_positions()
                
                error = np.absolute(gt_joint_pos - current_joint_pos)
                print(f"current joint pos: {current_joint_pos}")
                log_important(f"Joint pos error: {error}")
            
            is_continue = input("continue?(y/n)")
            while is_continue != "y" and is_continue != "n":
                # start = input("start?(y/n)")
                is_continue = input("continue?(y/n)")
            
            env.reset()
                
            if is_continue:
                save_list.append([gt_joint_pos, current_joint_pos])
                count += 1
            else:
                continue

    save_filepath = '/home/robots/sim_to_real/sim_to_real_trials_1.pkl'
    with open(save_filepath, "wb") as file:
        pickle.dump(save_list, file)
        log_success(f"saved {save_filepath}")

if __name__ == "__main__":
    run_experiment()