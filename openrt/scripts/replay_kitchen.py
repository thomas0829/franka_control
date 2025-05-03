import glob
import os
import time
import h5py
import pickle

import hydra
import imageio
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from robot.sim.vec_env.vec_env import make_env
from utils.experiment import hydra_to_dict
#from openrt.scripts.convert_np_to_hdf5 import normalize, unnormalize
from robot.wrappers.crop_wrapper import CropImageWrapper
from robot.wrappers.resize_wrapper import ResizeImageWrapper
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"

def normalize(arr, stats):
    min_val, max_val = stats["min"], stats["max"]
    return 2 * (arr - min_val) / (max_val - min_val) - 1


def unnormalize(arr, stats):
    min_val, max_val = stats["min"], stats["max"]
    return 0.5 * (arr + 1) * (max_val - min_val) + min_val

def replay_episode(demo, env):
    # stack data
    dic = {}
    obs_keys = demo[0].keys()
    for key in obs_keys:
        dic[key] = np.stack([d[key] for d in demo])
    # abs control
    actions = dic["lowdim_ee"][1:]

    THRESHOLD = 0.3
    actions[:, -1] = np.where(actions[:, -1] > THRESHOLD, 1, 0)

    demo_length = len(demo)
    action_interval = 20
    print("demo_length: ", demo_length) 
    for step_idx in tqdm(range(demo_length-1)):
        next_obs, reward, done, info = env.step_with_pose(actions[step_idx])

        # if step_idx % action_interval == 0:
        #     import pdb;pdb.set_trace()
        print("action: ", actions[step_idx])


# config_name="collect_demos_real" for real robot config_name="collect_demos_sim" for simulation
@hydra.main(
    config_path="../../configs/", config_name="collect_demos_real", version_base="1.1"
)
def run_experiment(cfg):

    # logdir = os.path.join(cfg.log.dir, cfg.exp_id)
    # os.makedirs(logdir, exist_ok=True)

    cfg.robot.max_path_length = cfg.max_episode_length
    print(cfg.robot.blocking_control, cfg.robot.control_hz)
    #assert cfg.robot.blocking_control==True and cfg.robot.control_hz<=1, "WARNING: please make sure to pass robot.blocking_control=true robot.control_hz=1 to run blocking control!"
    env = make_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        env_cfg_dict=hydra_to_dict(cfg.env) if "env" in cfg.keys() else None,
        seed=cfg.seed,
        device_id=0,
        verbose=True,
    )

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
    env.reset()

    demo_dir = "/home/prior/data_collection/data/38/pick_duster_npy/pick_duster_1/train/episode_1.npy"
    demo = np.load(demo_dir, allow_pickle=True)
    replay_episode(demo, env)


if __name__ == "__main__":
    run_experiment()




