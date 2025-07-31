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

def binary_prompt(name):
    bool_var = input(f"{name}?(y/n)")
    while bool_var not in ["y", "n"]:
        bool_var = input(f"{name}?(y/n)")
    return bool_var

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
    config_path="../../configs/", config_name="collect_demos_real_thinkpad", version_base="1.1"
)
def run_experiment(cfg):
    FLAGS(sys.argv)

    # configs
    log_config(f"language instruction: {cfg.language_instruction}")
    log_config(f"number of episodes: {cfg.episodes}")
    log_config(f"control hz: {cfg.robot.control_hz}")
    log_config(f"dataset name: {cfg.exp_id}")
    log_success("Resetting env")
    env = make_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        seed=cfg.seed,
        device_id=0,
        verbose=True,
    )


    # load .npz file
    import glob
    npz_file_paths = sorted(glob.glob("/home/robots/Downloads/system_identification/*.npy"))
    print(f"Found {len(npz_file_paths)} npy files.")
    for i, np_file_path in enumerate(npz_file_paths):
        print(f"[{i}] {os.path.basename(np_file_path)}")
    while True:
        index = input("select which one you would like to execute.")
        if index not in [str(i) for i in range(len(npz_file_paths))]:
            index = input("select which one you would like to execute.")
        demonstration_sim = np.load(npz_file_paths[int(index)])
        print(f"demonstration shape: {demonstration_sim.shape}")
        
        env = DataCollectionWrapper(
            env,
            language_instruction="system identification",
            fake_blocking=False,
            act_noise_std=cfg.act_noise_std,
            save_dir=npz_file_paths[int(index)].replace(".npy", "_real"),
        )

        oculus = VRController(pos_action_gain=10, rot_action_gain=4) # sensitivity 
        assert oculus.get_info()["controller_on"], "ERROR: oculus controller off"
        log_success("Oculus Connected")

        weight = ["1000"]
        num_trials = len(weight)
        

        for i in range(num_trials):
            # breakpoint()
            obs = env.reset(qpos=demonstration_sim[0][:7])
            while True:
                
                act = get_input_action(env, oculus, cfg)
                act[:6] = 0
                next_obs, rew, done, _ = env.step(act)

                print("act", act, "gripper: ", next_obs["lowdim_ee"][-1])
                info = oculus.get_info()
                if info["success"]:
                    env.reset_buffer()
                    break
            
            env.reset_buffer()
            for j, curr_qpos in enumerate(demonstration_sim):
                if j == 0:
                    continue
                else:
                    # curr_qpos[-1] = next_obs["lowdim_ee"][-1]
                    # curr_qpos[-1] = 
                    env.step_joint(curr_qpos)

            # path_name = input('Enter filename')
            path_name = npz_file_paths[int(index)].replace(".npy", f"_{weight[i]}_real.npy")
            env.save_buffer_joint(pickle_only=True, path_name=path_name)
            env.reset_buffer()

    # num_trials = 5
    # for i in range(num_trials):
    #     save = False
    #     while not save:
    #         obs = env.reset(qpos=demonstration_sim[0][:7])
    #         print("resetting environment...")
    #         start = binary_prompt(name="start")

    #         for i, curr_qpos in enumerate(demonstration_sim):
    #             if i == 0:
    #                 continue
    #             else:
    #                 env.step_joint(curr_qpos)
    #         save = binary_prompt(name="save")
    #         if save:
    #             path_name = input('Enter filename')
    #             env.save_buffer(pickle_only=True, path_name=path_name)
    #             print("saved")

if __name__ == "__main__":
    run_experiment()