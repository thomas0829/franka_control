import os
import time

import hydra
import numpy as np
from tqdm import tqdm

from robot.controllers.oculus import VRController, BimanualVRController
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




@hydra.main(
    config_path="../../configs/", config_name="collect_demos_real", version_base="1.1"
)
def run_experiment(cfg):
    FLAGS(sys.argv)
    logdir = os.path.join(cfg.log.dir, cfg.exp_id)
    os.makedirs(logdir, exist_ok=True)

    cfg.robot.max_path_length = cfg.max_episode_length
    assert cfg.robot.imgs, "ERROR: set robot.imgs=true to record image observations!"

    # create env
    env = make_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        seed=cfg.seed,
        device_id=0,
        verbose=True,
    )

    # camera_names = [k + "_rgb" for k in env.get_images().keys()]
    camera_names = [k for k in env.get_images().keys()]

    print(f"Camera names: {camera_names}")

    # crop image observations
    if cfg.aug.camera_crop is not None:
        env = CropImageWrapper(
            env,
            x_min=cfg.aug.camera_crop[0],
            x_max=cfg.aug.camera_crop[1],
            y_min=cfg.aug.camera_crop[2],
            y_max=cfg.aug.camera_crop[3],
            image_keys=[cn + "_rgb" for cn in camera_names],
            crop_render=True,
        )

    # resize image observations
    '''
    if cfg.aug.camera_resize is not None:
        env = ResizeImageWrapper(
            env,
            size=cfg.aug.camera_resize,
            image_keys=[cn + "_rgb" for cn in camera_names],
        )
    '''
    obs = env.reset()

    from datetime import date
    # creating the date object of today's date 
    todays_date = date.today() 

    savedir = f"{cfg.base_dir}/{todays_date.month}{todays_date.day}/{cfg.exp_id}/{cfg.split}"
    env = DataCollectionWrapper(
        env,
        language_instruction=cfg.language_instruction,
        fake_blocking=False,
        act_noise_std=cfg.act_noise_std,
        save_dir=savedir,
    )

    fig, ax = initialize_3d_plot()
    oculus = VRController(pos_action_gain=10, rot_action_gain=4) # sensitivity 
    # oculus = BimanualVRController(pos_action_gain=5)
    # oculus = VRController()
    assert oculus.get_info()["controller_on"], "ERROR: oculus controller off"
    print("Oculus Connected")

    n_traj = int(cfg.start_traj)
    env.traj_count = n_traj

    while n_traj < cfg.episodes:

        # reset w/o recording obs and w/o randomizing ee pos
        randomize_ee_on_reset = env.unwrapped._randomize_ee_on_reset
        env.unwrapped._set_randomize_ee_on_reset(0.0)
        env.unwrapped.reset()
        env.unwrapped._set_randomize_ee_on_reset(randomize_ee_on_reset)

        # make sure at least 1 camera is connected
        assert env.unwrapped._num_cameras > 0, "ERROR: not camera(s) connected!"

        print(f"Press 'A' to Start Collecting")
        # time to reset the scene
        while True:
            info = oculus.get_info()
            if info["success"]:
                # reset w/ recording obs after resetting the scene
                obs = env.reset()
                print("Start Collecting")
                time.sleep(1)
                break

        print(f"Press 'A' to Indicate SUCCESS, Press 'B' to Indicate FAILURE")

        obss = []
        acts = []

        # no-ops related variables
        first_no_ops_detected = True
        no_ops_start_time = 0
        no_ops_threshold = cfg.no_ops_threshold
        mode = cfg.mode
        no_ops_last_detected_time = cfg.no_ops_last_detected_time
        print("no_ops_threshold", no_ops_threshold)
        
        # lock rotation configs
        loc_rotation_start_time = 0
        lock_rotation_detected_time = cfg.lock_rotation_detected_time
        lock_rotation_first_detected = True
        
        print("mode", mode)
        for j in tqdm(
            range(cfg.max_episode_length),
            desc=f"Collecting Trajectory {n_traj}/{cfg.episodes}",
        ):

            # wait for controller input
            info = oculus.get_info()

            # lock rotation when not movement enabled
            if info["X"]:
                oculus.toggle_lock_rotation()
                print("lock rotation toggled")
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

            # # lock rotation if needed
            # if info["X"]:
            #     # first lock rotation detected
            #     if lock_rotation_first_detected:
            #         loc_rotation_start_time = time.time()
            #         lock_rotation_first_detected = False
            #     # lock rotation detected
            #     elif time.time() - loc_rotation_start_time >= lock_rotation_detected_time:
            #         oculus.toggle_lock_rotation()
            #         # reset lock rotation
            #         lock_rotation_first_detected = True
            #         loc_rotation_start_time = 0
            #     print("LOCK ROTATION DETECTED!!!!!!!!!!!!!!!!!!!!!")
            # # no lock rotation detected
            # else:
            #     lock_rotation_first_detected = True
            #     loc_rotation_start_time = 0
            #     print("no lock rotation detected")
                    
            
            # check if 'trigger' button is pressed
            if info["movement_enabled"]:
                    
                # prepare obs for oculus
                pose = env.unwrapped._robot.get_ee_pose()
                gripper = env.unwrapped._robot.get_gripper_position()
                
                # print("gripper", gripper)
                state = {
                    "robot_state": {
                        "cartesian_position": pose,
                        "gripper_position": gripper,
                    }
                }
                # qpos = env.unwrapped._robot.get_joint_positions()
                # print(f'qpos: {qpos}')
                # print(f'gipper: {gripper}')
                # vel_act, info = oculus.forward(state, include_info=True)
                vel_act, info =  oculus.forward(state, include_info=True, method="delta_action")

                update_3d_plot(ax, info["delta_action"][:3], info["target_pos_offset"][:3], info["robot_pos_offset"][:3])
                # update_3d_plot(ax, info["delta_action"][:3])
                
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

                # print(f"act: {act}")   
                # print("gripper", act[-1])
                # convert all actions to zero
                # act = np.zeros_like(act)   
                # import pdb; pdb.set_trace()

                # if oculus.vr_state["r"]["gripper"] > 0.5:
                if oculus.vr_state["gripper"] > 0.5:
                    act[-1] = 0.5
                else:
                    act[-1] = 0

                

                
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
                            print("No-ops count exceeded threshold")
                            break
                    # no-ops not detected (reset)
                    else:
                        first_no_ops_detected = True
                        no_ops_start_time = 0
                
                
                next_obs, rew, done, _ = env.step(act)
            
                
                # print("qpos", next_obs["lowdim_qpos"])
                # cv2.imshow('Real-time video', cv2.cvtColor(next_obs["215122255213_rgb"], cv2.COLOR_BGR2RGB))
                # cv2.imshow('Real-time video', cv2.cvtColor(next_obs[f"{camera_names[0]}_rgb"], cv2.COLOR_BGR2RGB))
                
                
                # update_3d_plot(ax, act[:3], act[3:6])
                
                
                # # Press 'q' on the keyboard to exit the loop
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                # emulate frequency in sim
                if cfg.robot.ip_address == None:
                    time.sleep(1 / cfg.robot.control_hz)
                    env.render()

                # # remove depth from observations
                # for cn in camera_names:
                #     if cn + "_depth" in obs:
                #         del obs[cn + "_depth"]

                obss.append(obs)
                acts.append(act)

                obs = next_obs

        # save trajectory if success
        if save:
            env.save_buffer()
            n_traj += 1
            print("SUCCESS")
        else:
            print("FAILURE")

    env.reset()

    print(f"Finished Collecting {n_traj} Trajectories")


if __name__ == "__main__":
    run_experiment()