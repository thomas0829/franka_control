import glob
import os
import time

import hydra

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange

from robot.wrappers.crop_wrapper import CropImageWrapper
from robot.wrappers.resize_wrapper import ResizeImageWrapper
from robot.robot_env import RobotEnv
from robot.sim.vec_env.vec_env import make_env
from utils.experiment import hydra_to_dict
import cv2
import robomimic.utils.file_utils as FileUtils
import h5py
from openrt.scripts.convert_np_to_hdf5 import *

import argparse
import imageio
from pathlib import Path


def unnormalize(arr, stats):
    min_val, max_val = stats["min"], stats["max"]
    return 0.5 * (arr + 1) * (max_val - min_val) + min_val


@hydra.main(config_path="../../configs/",
            config_name="collect_cube_sim",
            version_base="1.1")
def run_experiment(cfg):

    mode = "replay"  #["train", "test", "replay", "train_obs"]

    device = "cuda:0"
    ckpt_path = "/home/aurmr/robomimic_openrt/bc_trained_models/co_bc/20240613151543/models/model_epoch_40.pth"
    path_dir = "/media/aurmr/data/polymetis_franka/data/"
    # path_dir = "data"
    file = h5py.File(f"{path_dir}/{cfg.exp_id}/demos.hdf5", 'r')
    dataset = file["data"]
    import pickle
    with open(f"{path_dir}/{cfg.exp_id}/stats", 'rb') as file:
        # Load data from the file using pickle
        stats = pickle.load(file)["action"]
    if mode != "replay":
        policy, ckpt_dict = FileUtils.policy_from_checkpoint(
            ckpt_path=ckpt_path, device=device, verbose=True)

    logdir = os.path.join(cfg.log.dir, cfg.exp_id)
    os.makedirs(logdir, exist_ok=True)
    Path(f"{path_dir}/{cfg.exp_id}/evaluation").mkdir(exist_ok=True)
    Path(f"{path_dir}/{cfg.exp_id}/evaluation/image/").mkdir(exist_ok=True)
    Path(f"{path_dir}/{cfg.exp_id}/evaluation/pcd/").mkdir(exist_ok=True)
    cfg.robot.blocking_control = True

    if cfg.robot.blocking_control:
        cfg.robot.control_hz = 5

    # dataset_path = f"{path_dir}/{cfg.exp_id}/train"
    # file_names = glob.glob(f"{dataset_path}/episode_*.npy")
    # assert len(file_names) > 0, f"WARNING: no data in {dataset_path}!"

    env = make_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        env_cfg_dict=hydra_to_dict(cfg.env),
        seed=cfg.seed,
        device_id=0,
        verbose=True,
    )

    camera_names = env.unwrapped._robot.camera_names.copy()

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

    if cfg.aug.camera_resize is not None:
        env = ResizeImageWrapper(
            env,
            size=cfg.aug.camera_resize,
            image_keys=[cn + "_rgb" for cn in camera_names],
        )

    count = 0
    succ_count = 0

    for i in trange(200):

        # reset env
        obs = env.reset()
        if mode == "replay" or mode == "train" or mode == "train_obs":
            env.env.env.update_obj(dataset[f"demo_{i}"]["obs"]["obj_pose"][0][[
                0, 1, 2, 6, 3, 4, 5
            ]])

        # original trajectory
        traj0 = np.array(dataset[f"demo_{i}"]["actions"])
        # normalize trajectory
        traj0 = unnormalize(traj0, stats=stats)
        observations0 = dataset[f"demo_{i}"]["obs"]
        print(observations0["language_instruction"][0])

        for i in range(len(traj0)):

            obs["front_rgb"] = np.transpose(obs["front_rgb"], (2, 0, 1)) / 255

            pcd_obs = {}

            for key in obs.keys():
                pcd_obs[key] = obs[key][None]

            if mode == "train" or mode == "test":
                act = policy(obs)
                act = unnormalize(act, stats=stats)

            elif mode == "train_obs":

                obs0 = {}

                for key in ["front_rgb", "lang_embed", "lowdim_ee"]:
                    obs0[key] = observations0[key][i]

                    if key == "pcd":
                        obs0["pcd"] = np.transpose(obs0[key], (1, 0))

                obs0["front_rgb"] = np.transpose(obs0["front_rgb"],
                                                 (2, 0, 1)) / 255

                act = policy(obs0)
                act = unnormalize(act, stats=stats)

            elif mode == "replay":
                act = traj0[i]

            # act[-1] = traj0[i][-1]
            act[-1] = np.round(act[-1])
            next_obs, rew, done, _ = env.step(act)

            image = next_obs["front_rgb"]
            image = np.concatenate([image, observations0[f"front_rgb"][i]],
                                   axis=1)

            cv2.imshow('Real-time video', cv2.cvtColor(image,
                                                       cv2.COLOR_BGR2RGB))

            # Press 'q' on the keyboard to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            obs = next_obs
        count += 1

      
        if obs["obj_pose"][2] > 0.10:
            succ_count += 1
        print("success rate", succ_count / count)

        env.reset()

    env.reset()


if __name__ == "__main__":
    run_experiment()
