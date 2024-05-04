import os
import time

import hydra
import numpy as np
from tqdm import tqdm

from robot.controllers.oculus import VRController
from robot.crop_wrapper import CropImageWrapper
from robot.data_wrapper import DataCollectionWrapper
from robot.resize_wrapper import ResizeImageWrapper
from robot.sim.vec_env.vec_env import make_env
from utils.experiment import hydra_to_dict


@hydra.main(
    config_path="../../configs/", config_name="collect_demos_real", version_base="1.1"
)
def run_experiment(cfg):

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
    if cfg.aug.camera_resize is not None:
        env = ResizeImageWrapper(
            env,
            size=cfg.aug.camera_resize,
            image_keys=[cn + "_rgb" for cn in camera_names],
        )

    obs = env.reset()
    print("Observation keys", obs.keys())

    savedir = f"data/{cfg.exp_id}/{cfg.split}"
    env = DataCollectionWrapper(
        env,
        language_instruction=cfg.language_instruction,
        fake_blocking=False,
        act_noise_std=cfg.act_noise_std,
        save_dir=savedir,
    )

    oculus = VRController()
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
                break

        print(f"Press 'A' to Indicate SUCCESS, Press 'B' to Indicate FAILURE")

        obss = []
        acts = []

        for j in tqdm(
            range(cfg.max_episode_length),
            desc=f"Collecting Trajectory {n_traj}/{cfg.episodes}",
        ):

            # wait for controller input
            info = oculus.get_info()
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

                # prepare obs for oculus
                pose = env.unwrapped._robot.get_ee_pose()
                gripper = env.unwrapped._robot.get_gripper_position()
                state = {
                    "robot_state": {
                        "cartesian_position": pose,
                        "gripper_position": gripper,
                    }
                }
                vel_act = oculus.forward(state)

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

                next_obs, rew, done, _ = env.step(act)
                # emulate frequency in sim
                if cfg.robot.ip_address == None:
                    time.sleep(1 / cfg.robot.control_hz)
                    env.render()

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
