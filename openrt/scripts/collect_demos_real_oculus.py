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
        print(f"[DEBUG] Gripper CLOSED - act[-1] = {act[-1]}")
    else:
        act[-1] = 0
        print(f"[DEBUG] Gripper OPEN - act[-1] = {act[-1]}")
    
    return act


@hydra.main(
    config_path="../../configs/", config_name="collect_demos_real_thinkpad", version_base="1.1"
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
    # savedir = f"{cfg.base_dir}/date_{date.today().month}{date.today().day}/npy/{cfg.exp_id}/{cfg.split}"
    savedir = f"{cfg.base_dir}/date_{date.today().month}{date.today().day}/{cfg.exp_id}"
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
    
    env = DataCollectionWrapper(
        env,
        language_instruction=cfg.language_instruction,
        fake_blocking=False,
        act_noise_std=cfg.act_noise_std,
        save_dir=savedir,
    )
    obs = env.reset()
    log_success("Resetting env")
    
    # TODO: (yuquan) implement cropping and resizing
    # # crop image observations
    # if cfg.aug.camera_crop is not None:
    #     env = CropImageWrapper(
    #         env,
    #         x_min=cfg.aug.camera_crop[0],
    #         x_max=cfg.aug.camera_crop[1],
    #         y_min=cfg.aug.camera_crop[2],
    #         y_max=cfg.aug.camera_crop[3],
    #         image_keys=[cn + "_rgb" for cn in camera_names],
    #         crop_render=True,
    #     )
    # if cfg.aug.camera_resize is not None:
    #     env = ResizeImageWrapper(
    #         env,
    #         size=cfg.aug.camera_resize,
    #         image_keys=[cn + "_rgb" for cn in camera_names],
    #     )
    
    # TODO: (yuquan) better logging
    camera_names = [k for k in env.get_images().keys()]
    log_success(f"Initialized {len(camera_names)} camera(s): {camera_names}")
    # assert len(camera_names) == 3, "Make sure 3 cameras are connected!"
    
    
    # initialize oculus controller
    oculus = VRController(pos_action_gain=10, rot_action_gain=4) # sensitivity 
    assert oculus.get_info()["controller_on"], "ERROR: oculus controller off"
    log_success("Oculus Connected")

    # # visualize 3d plot
    # fig, ax = initialize_3d_plot()
        
    n_traj = int(cfg.start_traj)
    env.traj_count = n_traj
    while n_traj < cfg.episodes:

        # reset w/o recording obs and w/o randomizing ee pos
        randomize_ee_on_reset = env.unwrapped._randomize_ee_on_reset
        env.unwrapped._set_randomize_ee_on_reset(0.0)
        env.unwrapped.reset()
        env.unwrapped._set_randomize_ee_on_reset(randomize_ee_on_reset)

    #    # make sure at least 1 camera is connected
    #     assert env.unwrapped._num_cameras > 0, "ERROR: not camera(s) connected!"

        log_instruction("Press 'A' to Start Collecting")
        # time to reset the scene
        while True:
            info = oculus.get_info()
            if info["success"]:
                # reset w/ recording obs after resetting the scene
                obs = env.reset()
                log_instruction("Start Collecting")
                break

        log_instruction("Press 'A' to Indicate SUCCESS, Press 'B' to Indicate FAILURE")

        # no-ops related variables
        first_no_ops_detected = True
        no_ops_start_time = 0
        for j in tqdm(
            range(cfg.max_episode_length),
            desc=f"Collecting Trajectory {n_traj}/{cfg.episodes}",
        ):

            # wait for controller input
            info = oculus.get_info()

            # lock rotation when not movement enabled
            if info["X"]:
                oculus.toggle_lock_rotation()
                if oculus.lock_rotation:
                    log_success("Lock rotation enabled")
                else:
                    log_failure("Lock rotation disabled")
                time.sleep(0.1)
                    
            while (not info["success"] and not info["failure"]) and not info[
                "movement_enabled"
            ]:
                info = oculus.get_info()

            # press 'A' to indicate success
            save = False
            if info["success"]:
                save = True
                continue
            # press 'B' to indicate failure
            elif info["failure"]:
                continue
            
            # check if 'trigger' button is pressed
            if info["movement_enabled"]:

                act = get_input_action(env, oculus, cfg)
                
                # check if no-ops
                if cfg.mode == "standard":
                    act_norm = np.linalg.norm(act)
                    
                    # first no-ops detected
                    if act_norm < no_ops_threshold and first_no_ops_detected:
                        first_no_ops_detected = False
                        no_ops_start_time = time.time() # start time of no-ops
                    # no-ops detected
                    elif act_norm < no_ops_threshold and not first_no_ops_detected:
                        if time.time() - no_ops_start_time >= no_ops_last_detected_time:
                            log_failure(f"No operation for over {round(time.time() - no_ops_start_time, 2)} secs")
                            break
                    # no-ops not detected (reset)
                    else:
                        first_no_ops_detected = True
                        no_ops_start_time = 0
                
                next_obs, rew, done, _ = env.step(act)
                obs = next_obs
                
                
                qpos = env.unwrapped._robot.get_joint_positions()
                print("qpos: ", qpos)
                
                # COMMENTED OUT: Image visualization (requires cameras)
                # visual_img = None
                # for camera_name in camera_names:
                #     if visual_img is None:
                #         visual_img = obs[f"{camera_name}_rgb"]
                #     else:
                #         visual_img = np.concatenate(
                #             (visual_img, obs[f"{camera_name}_rgb"]), axis=1
                #         )
                        
                # cv2.imshow('Real-time video', cv2.cvtColor(visual_img, cv2.COLOR_BGR2RGB))
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

        # save trajectory if success
        if save:
            # env.save_buffer()
            env.save_buffer()
            n_traj += 1
            log_success("SUCCESS")
        else:
            log_failure("FAILURE")

    env.reset()
    log_success(f"Finished Collecting {n_traj} Trajectories")


if __name__ == "__main__":
    run_experiment()