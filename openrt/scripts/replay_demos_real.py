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


# config_name="collect_demos_real" for real robot config_name="collect_demos_sim" for simulation
@hydra.main(
    config_path="../../configs/", config_name="collect_demos_real", version_base="1.1"
)
def run_experiment(cfg):

    logdir = os.path.join(cfg.log.dir, cfg.exp_id)
    os.makedirs(logdir, exist_ok=True)

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

    #dataset_path = f"/media/marius/X9 Pro/0523_mujoco_data/{cfg.exp_id}/"
    dataset_path = f"/home/joel/projects/polymetis_franka/data/{cfg.exp_id}/"
    
    num_trajectories = 10

    file = h5py.File(os.path.join(dataset_path, "demos.hdf5"), 'r')
    dataset = file["data"]
    with open(os.path.join(dataset_path, "stats"),
              'rb') as file:
        stats = pickle.load(file)
    print(stats)

    for demo_key in list(dataset.keys())[:num_trajectories]:
        
        episode = dataset[demo_key]

        next_obs = env.reset()
        obs = next_obs

        obss = []
        acts = []

        actions = np.array(episode["actions"])
        actions = unnormalize(actions, stats=stats["action"])
        index = 0
        for act in tqdm(actions):
            import cv2
            
            # image = np.concatenate((episode["obs"]["front_rgb"][index], next_obs["832112071644_rgb"]), axis=1)
            # cv2.imshow('Real-time video', cv2.cvtColor(image,
            #                                            cv2.COLOR_BGR2RGB))

            start_time = time.time()

            #print(act[3:6])
            #act[[3,4,5]] = act[[4,3,5]]
            #act[4] = -act[4]
            #act[3] = -act[3]
            next_obs, rew, done, _ = env.step(act)
         
            # writer.append_data(image)
            index += 1
            # Press 'q' on the keyboard to exit the loop
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            obss.append(obs)
            acts.append(act)
            obs = next_obs

            print(f"Act {act} \nTook {np.around(time.time() - start_time, 3)}s")
        env.reset()

        # visualize traj
        img_obs = np.stack([obs[camera_names[0] + "_rgb"] for obs in obss])
        img_keys = [key for key in dataset[demo_key]["obs"].keys() if "rgb" in key]
        img_demo = np.array(dataset[demo_key]["obs"][img_keys[0]])
        '''
        imageio.mimsave(os.path.join(logdir, f"demo_replay_{demo_key}.mp4"), np.concatenate((img_demo, img_obs), axis=2))

        # plot difference between demo and replay
        ee_obs = np.stack([obs["lowdim_ee"] for obs in obss])
        ee_demo = np.array(dataset[demo_key]["obs"]["lowdim_ee"])

        labels = ["x", "y", "z", "r", "p", "y"]
        colors = ["tab:orange", "tab:blue", "tab:green", "tab:red", "tab:purple", "tab:brown"]
        n = 3
        for j, (l,c) in enumerate(zip(labels[:n], colors[:n])):
            plt.plot(ee_demo [:,j], color=c, label=f"{l} demo")
            plt.plot(ee_obs[:,j], color=c, linestyle="dashed", label=f"{l} replay")
        plt.plot(ee_demo [:,-1], color="red", label=f"gripper demo")
        plt.plot(ee_obs[:,-1], color="red", linestyle="dashed", label=f"gripper replay")
        plt.legend()
        plt.savefig(os.path.join(logdir, f"poses_{demo_key}.png"))
        plt.close()
        '''

    env.reset()


if __name__ == "__main__":
    run_experiment()
